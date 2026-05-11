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

    removeCols = [
        col for col in tokenized["train"].column_names
        if col not in {"input_ids", "attention_mask", "labels"}
    ]
    tokenized = tokenized.remove_columns(removeCols)

    return tokenized, tokenizer, TASK_TO_NUM_LABELS[task]


def compute_metrics(evalPred):
    logits, labels = evalPred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def build_training_args(cfg: dict, runName: str) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(Path("checkpoints") / runName),
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
        "run_name": runName,
        "fp16": bool(torch.cuda.is_available() and cfg.get("fp16", True)),
        "seed": cfg.get("seed", 0),
        "data_seed": cfg.get("seed", 0),
    }

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"

    supportedKwargs = {k: v for k, v in kwargs.items() if k in params}
    ignoredKwargs = sorted(set(kwargs) - set(supportedKwargs))
    if ignoredKwargs:
        print("[train] Ignoring unsupported TrainingArguments: " + str(ignoredKwargs))

    return TrainingArguments(**supportedKwargs)


def build_optimizer(model, cfg: dict):
    noDecay = ["bias", "LayerNorm.weight"]
    useLoraPlus = cfg.get("mode") == "lora" and cfg.get("lora_plus", False)
    lrDefault = cfg["learning_rate"]
    lrA = cfg.get("lora_plus_lr_A", lrDefault) if useLoraPlus else lrDefault
    lrB = lrA * cfg.get("lora_plus_ratio", 16.0) if useLoraPlus else lrDefault
    weightDecay = cfg.get("weight_decay", 0.01)

    from collections import defaultdict
    buckets = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = lrB if useLoraPlus and "loraB" in name else lrA
        wd = 0.0 if any(nd in name for nd in noDecay) else weightDecay
        buckets[(lr, wd)].append(param)

    groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in buckets.items()
        if params
    ]
    return AdamW(groups, lr=lrDefault)


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] Device: " + str(device))
    set_seed(cfg.get("seed", 0))

    tokenized, tokenizer, numLabels = get_tokenized_datasets(cfg)
    print(
        "[data] Task=" + str(cfg["task"]) + " | Train=" + format(len(tokenized["train"]), ",") +
        " | Val=" + format(len(tokenized["validation"]), ",") + " | Labels=" + str(numLabels)
    )

    model = build_model(
        numLabels=numLabels,
        modelName=cfg["model_name"],
        mode=cfg["mode"],
        rank=cfg.get("rank", 8),
        alpha=cfg.get("alpha", 8.0),
        dropout=cfg.get("lora_dropout", 0.0),
        targetModules=cfg.get("target_modules", ["query", "value"]),
        loraInit=cfg.get("lora_init", "paper"),
        loraMergeWeights=cfg.get("lora_merge_weights", True),
        loraTrainBias=cfg.get("lora_train_bias", "none"),
    ).to(device)

    trainable, total = count_parameters(model)

    runName = cfg.get("run_name", cfg["mode"] + "_r" + str(cfg.get("rank", "full")))
    os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "lora-replication")
    trainingArgs = build_training_args(cfg, runName)
    optimizer = build_optimizer(model, cfg)

    trainer = Trainer(
        model=model,
        args=trainingArgs,
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
    elapsedSec = time.time() - t0
    gpuMemoryMb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    metricsEval = trainer.evaluate()
    bestValAcc = float(
        trainer.state.best_metric
        if trainer.state.best_metric is not None
        else metricsEval.get("eval_accuracy", 0.0)
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        lora_state_dict(model) if cfg["mode"] == "lora" else model.state_dict(),
        "checkpoints/" + runName + "_best.pt",
    )

    print("\nBest val accuracy: " + format(bestValAcc, ".4f"))
    targetAcc = cfg.get("target_accuracy", None)
    if targetAcc is not None and bestValAcc < targetAcc:
        print(
            "[train] WARNING: best_val_accuracy=" + format(bestValAcc, ".4f") +
            " is below target_accuracy=" + format(targetAcc, ".4f") + "."
        )
    wandb.summary["best_val_accuracy"] = bestValAcc
    wandb.finish()

    repoRoot = Path(__file__).resolve().parents[1]
    logsDir = repoRoot / "results" / "logs"
    logsDir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_name": runName,
        "task": cfg["task"],
        "mode": cfg["mode"],
        "target_modules": ",".join(cfg.get("target_modules", [])),
        "rank": cfg.get("rank", None),
        "alpha": cfg.get("alpha", None),
        "val_accuracy": bestValAcc,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg.get("weight_decay", 0.0),
        "warmup_ratio": cfg.get("warmup_ratio", 0.0),
        "gradient_accumulation_steps": max(1, int(cfg.get("gradient_accumulation_steps", 1))),
        "seed": cfg.get("seed", 0),
        "elapsed_sec": elapsedSec,
        "gpu_memory_mb": gpuMemoryMb,
        "log_history": trainer.state.log_history,
    }
    with open(logsDir / (runName + ".json"), "w") as f:
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

    train(cfg)
