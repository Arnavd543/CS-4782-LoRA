"""
train.py
--------
Training loop supporting both full fine-tuning and LoRA fine-tuning.
Logs metrics to Weights & Biases and prints GPU memory usage each epoch.

Usage:
    python train.py --config configs/lora_r8.yaml
    python train.py --config configs/baseline.yaml
    python train.py --config configs/lora_r8.yaml --rank 4 --run_name lora_r4
"""

import argparse
import json
import os
import random
import sys
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data import get_dataloaders
from model import build_model
from evaluate import evaluate, gpu_memory_mb
from lora import count_parameters, lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--rank",     type=int,   default=None, help="Override rank")
    parser.add_argument("--alpha",    type=float, default=None, help="Override alpha")
    parser.add_argument("--run_name", type=str,   default=None, help="Override run name")
    parser.add_argument("--epochs",   type=int,   default=None, help="Override epochs")
    parser.add_argument("--target_modules", type=str, default=None, help="Comma-separated target modules")
    parser.add_argument("--lora_plus", action="store_true", help="Enable LoRA+")
    parser.add_argument("--lora_dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--lora_init", type=str, default=None, choices=["microsoft", "paper"], help="LoRA init style")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Override grad accumulation")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    set_seed(cfg.get("seed", 0))

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, num_labels = get_dataloaders(
        task=cfg["task"],
        model_name=cfg["model_name"],
        max_length=cfg.get("max_length", 128),
        batch_size=cfg["batch_size"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_labels=num_labels,
        model_name=cfg["model_name"],
        mode=cfg["mode"],
        rank=cfg.get("rank", 8),
        alpha=cfg.get("alpha", 8.0),
        dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("target_modules", ["query", "value"]),
        lora_init=cfg.get("lora_init", "paper"),
        lora_merge_weights=cfg.get("lora_merge_weights", True),
        lora_train_bias=cfg.get("lora_train_bias", "none"),
    ).to(device)

    trainable, total = count_parameters(model)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    NO_DECAY = ["bias", "LayerNorm.weight"]
    use_lora_plus = cfg.get("mode") == "lora" and cfg.get("lora_plus", False)
    lr_default = cfg["learning_rate"]
    lr_A = cfg.get("lora_plus_lr_A", lr_default) if use_lora_plus else lr_default
    lr_B = lr_A * cfg.get("lora_plus_ratio", 16.0) if use_lora_plus else lr_default
    wd   = cfg.get("weight_decay", 0.01)

    from collections import defaultdict
    buckets = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_wd = 0.0 if any(nd in name for nd in NO_DECAY) else wd
        if use_lora_plus:
            lr = lr_B if "lora_B" in name else lr_A
        else:
            lr = lr_default
        buckets[(lr, param_wd)].append(param)

    groups = [{"params": p, "lr": lr, "weight_decay": w}
              for (lr, w), p in buckets.items() if p]
    optimizer = AdamW(groups, lr=lr_default)

    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    updates_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    total_steps = updates_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.06))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    run_name = cfg.get("run_name", f"{cfg['mode']}_r{cfg.get('rank','full')}")
    wandb.init(
        project=cfg.get("wandb_project", "lora-replication"),
        name=run_name,
        config={**cfg, "trainable_params": trainable, "total_params": total},
    )
    wandb.watch(model, log=None)

    # ── Training loop ─────────────────────────────────────────────────────────
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss if outputs.loss is not None else loss_fn(
                outputs.logits, batch["labels"]
            )
            total_loss += loss.item()

            (loss / grad_accum).backward()

            should_step = (step % grad_accum == 0) or (step == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        mem_mb = gpu_memory_mb(device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} "
              f"| gpu={mem_mb:.0f}MB | t={elapsed:.1f}s")

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "gpu_memory_mb": mem_mb,
            "trainable_params": trainable,
            "trainable_pct": 100 * trainable / total,
            "learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else lr_default,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                lora_state_dict(model) if cfg["mode"] == "lora" else model.state_dict(),
                f"checkpoints/{run_name}_best.pt",
            )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    target_acc = cfg.get("target_accuracy", None)
    if target_acc is not None and best_val_acc < target_acc:
        print(
            f"[train] WARNING: best_val_accuracy={best_val_acc:.4f} is below target_accuracy={target_acc:.4f}."
        )
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.finish()

    # Save metrics to JSON for analysis script
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "run_name": run_name,
        "mode": cfg["mode"],
        "target_modules": ",".join(cfg.get("target_modules", [])),
        "rank": cfg.get("rank", None),
        "alpha": cfg.get("alpha", None),
        "val_accuracy": best_val_acc,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg.get("weight_decay", 0.0),
        "warmup_ratio": cfg.get("warmup_ratio", 0.0),
        "gradient_accumulation_steps": grad_accum,
        "seed": cfg.get("seed", 0),
    }
    with open(logs_dir / f"{run_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.rank is not None:
        cfg["rank"] = args.rank
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.target_modules is not None:
        cfg["target_modules"] = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if args.lora_plus:
        cfg["lora_plus"] = True
    if args.lora_dropout is not None:
        cfg["lora_dropout"] = args.lora_dropout
    if args.lora_init is not None:
        cfg["lora_init"] = args.lora_init
    if args.gradient_accumulation_steps is not None:
        cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    
    train(cfg)
