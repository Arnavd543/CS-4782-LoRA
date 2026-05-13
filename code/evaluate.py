# Standalone evaluation utility (not part of the main training/analysis pipeline).
# The notebook flow uses train.py -> analyze.py -> forgetting.py and does not invoke this script.
# This file is provided for ad-hoc post-hoc evaluation of a single checkpoint:
#   python evaluate.py --checkpoint checkpoints/lora_r8_best.pt --config configs/lora_r8.yaml

import argparse
import os
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))


def evaluate(model, dataloader, device) -> float:
    # Run model over dataloader and return accuracy in [0, 1].
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
    # Return current GPU memory allocated in MB (0.0 on CPU).
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024 ** 2


def print_param_report(model) -> None:
    # Print a breakdown of trainable vs frozen parameters.
    total, trainable = 0, 0
    print("\n" + format("Layer", "<60") + format("Params", ">10") + format("Trainable", ">10"))
    print("-" * 82)
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            flag = "v"
        else:
            flag = " "
        print(flag + " " + format(name, "<58") + format(n, ">10,"))
    print("-" * 82)
    print("  " + format("TOTAL", "<58") + format(total, ">10,"))
    print("  " + format("TRAINABLE", "<58") + format(trainable, ">10,") +
          "  (" + format(100 * trainable / total, ".4f") + "%)\n")


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
    print("Validation accuracy: " + format(acc, ".4f"))
