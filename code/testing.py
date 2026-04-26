"""
testing.py
----------
Experiment driver for the LoRA replication project.

This script runs two sets of experiments and populates `results/` with
CSV tables and plots:

1. Rank sweep vs full fine-tuning.
2. LoRA target-module comparison at a fixed low rank.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import yaml
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data import get_dataloaders
from evaluate import evaluate, gpu_memory_mb
from lora import count_parameters, lora_state_dict
from model import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def ensure_dirs():
    for path in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    if cfg.get("mode") == "lora" and cfg.get("lora_plus", False):
        from lora import lora_plus_param_groups

        param_groups = lora_plus_param_groups(
            model,
            lr_A=cfg["learning_rate"],
            lr_B_multiplier=cfg.get("lora_plus_ratio", 16.0),
        )
        return AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.01))

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    return AdamW(trainable_params, lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.01))


def train_one_run(
    cfg: Dict[str, Any],
    run_name: str,
    target_modules: Optional[List[str]] = None,
    override_rank: Optional[int] = None,
    override_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(cfg)
    if target_modules is not None:
        cfg["target_modules"] = target_modules
    if override_rank is not None:
        cfg["rank"] = override_rank
    if override_alpha is not None:
        cfg["alpha"] = override_alpha

    train_loader, val_loader, num_labels = get_dataloaders(
        task=cfg["task"],
        model_name=cfg["model_name"],
        max_length=cfg.get("max_length", 128),
        batch_size=cfg["batch_size"],
    )

    model = build_model(
        num_labels=num_labels,
        model_name=cfg["model_name"],
        mode=cfg["mode"],
        rank=cfg.get("rank", 8),
        alpha=cfg.get("alpha", 8.0),
        dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("target_modules", ["query", "value"]),
    ).to(device)

    trainable, total = count_parameters(model)
    optimizer = build_optimizer(model, cfg)

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.06))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    best_metrics: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss if outputs.loss is not None else loss_fn(outputs.logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        mem_mb = gpu_memory_mb(device)
        elapsed = time.time() - start

        epoch_metrics = {
            "run_name": run_name,
            "mode": cfg["mode"],
            "target_modules": ",".join(cfg.get("target_modules", [])),
            "rank": cfg.get("rank", None),
            "alpha": cfg.get("alpha", None),
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total if total else 0.0,
            "gpu_memory_mb": mem_mb,
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(epoch_metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CHECKPOINTS_DIR / f"{run_name}_best.pt"
            state = lora_state_dict(model) if cfg["mode"] == "lora" else model.state_dict()
            torch.save(state, checkpoint_path)
            best_metrics = dict(epoch_metrics, checkpoint=str(checkpoint_path))

        print(
            f"[{run_name}] epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f} "
            f"gpu={mem_mb:.0f}MB elapsed={elapsed:.1f}s"
        )

    if best_metrics is None:
        best_metrics = history[-1]
        best_metrics["checkpoint"] = str(CHECKPOINTS_DIR / f"{run_name}_final.pt")
        torch.save(model.state_dict(), best_metrics["checkpoint"])

    best_metrics["history"] = history
    return best_metrics


def run_rank_sweep(
    baseline_cfg_path: Path,
    lora_cfg: Dict[str, Any],
    ranks: List[int],
    target_modules: List[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    baseline_cfg = load_config(baseline_cfg_path)
    baseline_cfg.setdefault("task", lora_cfg.get("task", baseline_cfg.get("task")))
    baseline_cfg.setdefault("model_name", lora_cfg.get("model_name", baseline_cfg.get("model_name")))
    baseline_cfg["epochs"] = lora_cfg.get("epochs", baseline_cfg.get("epochs"))
    baseline_cfg["batch_size"] = lora_cfg.get("batch_size", baseline_cfg.get("batch_size"))

    baseline_metrics = train_one_run(baseline_cfg, run_name="baseline_full")
    results.append(baseline_metrics)

    for rank in ranks:
        run_cfg = dict(lora_cfg)
        run_cfg["mode"] = "lora"
        run_cfg["rank"] = rank
        run_cfg["alpha"] = rank
        run_cfg["target_modules"] = target_modules
        run_name = f"lora_rank_{rank}"
        metrics = train_one_run(run_cfg, run_name=run_name)
        results.append(metrics)

    return results


def run_module_comparison(
    cfg: Dict[str, Any],
    rank: int,
    module_combinations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for combo in module_combinations:
        run_cfg = dict(cfg)
        run_cfg["mode"] = "lora"
        run_cfg["rank"] = rank
        run_cfg["alpha"] = rank
        run_cfg["target_modules"] = combo["targets"]
        run_name = f"lora_module_{combo['name'].replace(' ', '_')}"
        metrics = train_one_run(run_cfg, run_name=run_name)
        metrics["combo_name"] = combo["name"]
        results.append(metrics)
    return results


def plot_rank_sweep(rows: List[Dict[str, Any]], path: Path):
    ranks = [int(row["rank"]) if row["mode"] == "lora" else 0 for row in rows]
    acc = [row["val_accuracy"] for row in rows]
    params = [row["trainable_params"] for row in rows]
    labels = [row["run_name"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot([0] + [r for r in ranks if r != 0], [rows[0]["val_accuracy"]] + [row["val_accuracy"] for row in rows if row["mode"] == "lora"], marker="o")
    plt.xticks([0] + [r for r in ranks if r != 0], ["full"] + [str(r) for r in ranks if r != 0])
    plt.xlabel("LoRA rank")
    plt.ylabel("Validation accuracy")
    plt.title("LoRA rank sweep vs full fine-tuning")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot([0] + [r for r in ranks if r != 0], [rows[0]["trainable_params"]] + [row["trainable_params"] for row in rows if row["mode"] == "lora"], marker="o")
    plt.xticks([0] + [r for r in ranks if r != 0], ["full"] + [str(r) for r in ranks if r != 0])
    plt.xlabel("LoRA rank")
    plt.ylabel("Trainable parameters")
    plt.title("Trainable parameters by rank")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.with_name(path.stem + "_params.png"))
    plt.close()


def plot_module_comparison(rows: List[Dict[str, Any]], path: Path):
    labels = [row["combo_name"] for row in rows]
    acc = [row["val_accuracy"] for row in rows]
    params = [row["trainable_params"] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, acc, color="tab:blue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation accuracy")
    plt.title("LoRA target-module comparison")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, params, color="tab:orange")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Trainable parameters")
    plt.title("Trainable parameters by target-module combination")
    plt.tight_layout()
    plt.savefig(path.with_name(path.stem + "_params.png"))
    plt.close()


def main(args: argparse.Namespace):
    ensure_dirs()
    lora_cfg = load_config(Path(args.config))
    baseline_cfg_path = Path(__file__).resolve().parent / "configs" / "baseline.yaml"

    rank_sweep_ranks = args.ranks or [1, 2, 4, 8, 16, 32]
    module_combinations = [
        {"name": "Wq", "targets": ["query"]},
        {"name": "Wk", "targets": ["key"]},
        {"name": "Wv", "targets": ["value"]},
        {"name": "Q+V", "targets": ["query", "value"]},
        {"name": "QKV", "targets": ["query", "key", "value"]},
        {"name": "Attention_out", "targets": ["attention.output.dense"]},
        {"name": "FFN_up", "targets": ["intermediate.dense"]},
        {"name": "FFN_down", "targets": ["output.dense"]},
    ]

    print("[testing] Running rank sweep experiments...")
    rank_results = run_rank_sweep(baseline_cfg_path, lora_cfg, rank_sweep_ranks, target_modules=["query", "value"])
    save_json(RESULTS_DIR / "rank_sweep_results.json", rank_results)
    fields = [
        "run_name",
        "mode",
        "target_modules",
        "rank",
        "alpha",
        "val_accuracy",
        "trainable_params",
        "total_params",
        "trainable_pct",
        "gpu_memory_mb",
        "elapsed_sec",
        "checkpoint",
    ]
    save_csv(TABLES_DIR / "rank_sweep.csv", rank_results, fields)
    plot_rank_sweep(rank_results, FIGURES_DIR / "rank_sweep_accuracy.png")

    print("[testing] Running module comparison experiments...")
    module_results = run_module_comparison(lora_cfg, args.module_rank or 4, module_combinations)
    save_json(RESULTS_DIR / "module_comparison_results.json", module_results)
    module_fields = [
        "run_name",
        "combo_name",
        "mode",
        "target_modules",
        "rank",
        "alpha",
        "val_accuracy",
        "trainable_params",
        "total_params",
        "trainable_pct",
        "gpu_memory_mb",
        "elapsed_sec",
        "checkpoint",
    ]
    save_csv(TABLES_DIR / "module_comparison.csv", module_results, module_fields)
    plot_module_comparison(module_results, FIGURES_DIR / "module_comparison_accuracy.png")

    print("[testing] Finished. Results written to:")
    print(f"  {TABLES_DIR}")
    print(f"  {FIGURES_DIR}")
    print(f"  {LOGS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA experiment families and export results.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_r8.yaml",
        help="Path to a LoRA config YAML file.",
    )
    parser.add_argument(
        "--module_rank",
        type=int,
        default=4,
        help="Rank to use for the target-module comparison experiments.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="*",
        help="Optional list of ranks to sweep for the rank sweep experiment.",
    )
    args = parser.parse_args()
    main(args)
