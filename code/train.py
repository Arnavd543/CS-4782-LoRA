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
import os
import sys
import time
import yaml

import torch
import wandb
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data import get_dataloaders
from model import build_model
from evaluate import evaluate, gpu_memory_mb
from lora import lora_plus_param_groups, count_parameters, lora_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--rank",     type=int,   default=None, help="Override rank from config")
    parser.add_argument("--run_name", type=str,   default=None, help="Override W&B run name")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

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
    ).to(device)

    trainable, total = count_parameters(model)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    use_lora_plus = cfg.get("lora_plus", False) and cfg["mode"] == "lora"

    if use_lora_plus:
        param_groups = lora_plus_param_groups(
            model,
            lr_A=cfg["learning_rate"],
            lr_B_multiplier=cfg.get("lora_plus_ratio", 16.0),
        )
        optimizer = AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.01))
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 0.01),
        )

    total_steps = len(train_loader) * cfg["epochs"]
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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss if outputs.loss is not None else loss_fn(
                outputs.logits, batch["labels"]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

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
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                lora_state_dict(model) if cfg["mode"] == "lora" else model.state_dict(),
                f"checkpoints/{run_name}_best.pt",
            )

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.rank is not None:
        cfg["rank"] = args.rank
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    train(cfg)
