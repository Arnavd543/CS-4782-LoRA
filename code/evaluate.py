"""
evaluate.py
-----------
Evaluation utilities: accuracy over a DataLoader, trainable parameter
counting, and GPU memory reporting.

Also doubles as a standalone script:
    python evaluate.py --checkpoint checkpoints/lora_r8_best.pt \
                       --config configs/lora_r8.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))


def evaluate(model, dataloader, device) -> float:
    """
    Run model over `dataloader` and return accuracy.

    Args:
        model:      nn.Module in eval mode (function sets eval internally).
        dataloader: validation DataLoader.
        device:     torch.device.

    Returns:
        accuracy as a float in [0, 1].
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    model.train()
    return correct / total


def gpu_memory_mb(device) -> float:
    """Return current GPU memory allocated in MB (0.0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024 ** 2


def print_param_report(model) -> None:
    """Print a breakdown of trainable vs frozen parameters."""
    total, trainable = 0, 0
    print(f"\n{'Layer':<60} {'Params':>10} {'Trainable':>10}")
    print("-" * 82)
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            flag = "✓"
        else:
            flag = " "
        print(f"{flag} {name:<58} {n:>10,}")
    print("-" * 82)
    print(f"  {'TOTAL':<58} {total:>10,}")
    print(f"  {'TRAINABLE':<58} {trainable:>10,}  ({100*trainable/total:.4f}%)\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     required=True)
    return parser.parse_args()


if __name__ == "__main__":
    from data import get_dataloaders
    from model import build_model

    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, num_labels = get_dataloaders(
        task=cfg["task"],
        model_name=cfg["model_name"],
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
        lora_init=cfg.get("lora_init", "paper"),
        lora_merge_weights=cfg.get("lora_merge_weights", True),
        lora_train_bias=cfg.get("lora_train_bias", "none"),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)

    print_param_report(model)
    acc = evaluate(model, val_loader, device)
    print(f"Validation accuracy: {acc:.4f}")
