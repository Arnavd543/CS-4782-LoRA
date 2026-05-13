# Training loop for full fine-tuning and LoRA fine-tuning.
# Usage:
#   python train.py --config configs/lora_r8.yaml
#   python train.py --config configs/baseline.yaml
#   python train.py --config configs/lora_r8.yaml --rank 4 --run_name lora_r4

import argparse
import inspect
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
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(__file__))
from data import TASK_TO_KEYS, TASK_TO_NUM_LABELS
from model import build_model
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
    parser.add_argument("--task", type=str, default=None, choices=["sst2", "mrpc"], help="Override task")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Directory for *_best.pt and Trainer outputs")
    parser.add_argument("--results_dir", type=str, default=None, help="Root results dir; logs/ subdir is used")
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


def get_tokenized_datasets(cfg: dict):
    task = cfg["task"]
    if task not in TASK_TO_KEYS:
        raise ValueError(f"Unsupported task '{task}'. Choose from {list(TASK_TO_KEYS)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    raw = load_dataset("glue", task)
    key1, key2 = TASK_TO_KEYS[task]

    def tokenize(batch):
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])
        return tokenizer(*texts, truncation=True, max_length=cfg.get("max_length", 128))

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    remove_cols = [
        col for col in tokenized["train"].column_names
        if col not in {"input_ids", "attention_mask", "labels"}
    ]
    tokenized = tokenized.remove_columns(remove_cols)

    return tokenized, tokenizer, TASK_TO_NUM_LABELS[task]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def build_training_args(cfg: dict, run_name: str, checkpoints_dir: Path) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(checkpoints_dir / run_name),
        "overwrite_output_dir": True,
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg.get("eval_batch_size", 64),
        "num_train_epochs": cfg["epochs"],
        "weight_decay": cfg.get("weight_decay", 0.01),
        "warmup_ratio": cfg.get("warmup_ratio", 0.06),
        "gradient_accumulation_steps": max(1, int(cfg.get("gradient_accumulation_steps", 1))),
        "logging_steps": cfg.get("logging_steps", 50),
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "report_to": ["wandb"],
        "run_name": run_name,
        "fp16": bool(torch.cuda.is_available() and cfg.get("fp16", True)),
        "seed": cfg.get("seed", 0),
        "data_seed": cfg.get("seed", 0),
    }

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"

    supported_kwargs = {k: v for k, v in kwargs.items() if k in params}
    ignored_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if ignored_kwargs:
        print("[train] Ignoring unsupported TrainingArguments: " + str(ignored_kwargs))

    return TrainingArguments(**supported_kwargs)


def build_optimizer(model, cfg: dict):
    no_decay = ["bias", "LayerNorm.weight"]
    use_lora_plus = cfg.get("mode") == "lora" and cfg.get("lora_plus", False)
    lr_default = cfg["learning_rate"]
    lr_a = cfg.get("lora_plus_lr_A", lr_default) if use_lora_plus else lr_default
    lr_b = lr_a * cfg.get("lora_plus_ratio", 16.0) if use_lora_plus else lr_default
    weight_decay = cfg.get("weight_decay", 0.01)

    from collections import defaultdict
    buckets = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = lr_b if use_lora_plus and "loraB" in name else lr_a
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        buckets[(lr, wd)].append(param)

    groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in buckets.items()
        if params
    ]
    return AdamW(groups, lr=lr_default)


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] Device: " + str(device))
    set_seed(cfg.get("seed", 0))

    repo_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = Path(cfg.get("checkpoints_dir") or (repo_root / "checkpoints"))
    results_dir = Path(cfg.get("results_dir") or (repo_root / "results"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print("[train] checkpoints_dir = " + str(checkpoints_dir))
    print("[train] results_dir     = " + str(results_dir))

    tokenized, tokenizer, num_labels = get_tokenized_datasets(cfg)
    print(
        "[data] Task=" + str(cfg["task"]) + " | Train=" + format(len(tokenized["train"]), ",") +
        " | Val=" + format(len(tokenized["validation"]), ",") + " | Labels=" + str(num_labels)
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

    trainable, total = count_parameters(model)

    run_name = cfg.get("run_name", cfg["mode"] + "_r" + str(cfg.get("rank", "full")))
    os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "lora-replication")
    training_args = build_training_args(cfg, run_name, checkpoints_dir)
    optimizer = build_optimizer(model, cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed_sec = time.time() - t0
    gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    metrics_eval = trainer.evaluate()
    best_val_acc = float(
        trainer.state.best_metric
        if trainer.state.best_metric is not None
        else metrics_eval.get("eval_accuracy", 0.0)
    )

    torch.save(
        lora_state_dict(model) if cfg["mode"] == "lora" else model.state_dict(),
        str(checkpoints_dir / (run_name + "_best.pt")),
    )

    print("\nBest val accuracy: " + format(best_val_acc, ".4f"))
    target_acc = cfg.get("target_accuracy", None)
    if target_acc is not None and best_val_acc < target_acc:
        print(
            "[train] WARNING: best_val_accuracy=" + format(best_val_acc, ".4f") +
            " is below target_accuracy=" + format(target_acc, ".4f") + "."
        )
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.finish()

    metrics = {
        "run_name": run_name,
        "task": cfg["task"],
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
        "gradient_accumulation_steps": max(1, int(cfg.get("gradient_accumulation_steps", 1))),
        "seed": cfg.get("seed", 0),
        "elapsed_sec": elapsed_sec,
        "gpu_memory_mb": gpu_memory_mb,
        "log_history": trainer.state.log_history,
    }
    with open(logs_dir / (run_name + ".json"), "w") as f:
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
    if args.task is not None:
        cfg["task"] = args.task
        cfg["num_labels"] = 2
    if args.checkpoints_dir is not None:
        cfg["checkpoints_dir"] = args.checkpoints_dir
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir

    train(cfg)
