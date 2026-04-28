import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"

BASELINE_DIR = RESULTS_DIR / "baseline"
RANK_SWEEP_DIR = RESULTS_DIR / "rank_sweep"
MODULE_CMP_DIR = RESULTS_DIR / "module_comparison"
EXTENSIONS_DIR = RESULTS_DIR / "extensions"


def ensure_dirs():
    for path in [BASELINE_DIR, RANK_SWEEP_DIR, MODULE_CMP_DIR, EXTENSIONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "figures").mkdir(exist_ok=True)


def load_logs() -> Dict[str, Dict[str, Any]]:
    logs = {}
    if not LOGS_DIR.exists():
        print(f"No logs directory found at {LOGS_DIR}")
        return logs
    for p in LOGS_DIR.glob("*.json"):
        with open(p, "r") as f:
            data = json.load(f)
            logs[data["run_name"]] = data
    return logs


def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_baseline_comparison(full_ft: Dict[str, Any], lora_r8: Dict[str, Any], path: Path):
    labels = ["Full Fine-Tuning", "LoRA (r=8)"]
    accs = [full_ft["val_accuracy"], lora_r8["val_accuracy"]]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, accs, color=["tab:blue", "tab:orange"])
    plt.ylabel("Validation accuracy")
    plt.title("Baseline Comparison")
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_param_efficiency(full_ft: Dict[str, Any], lora_r8: Dict[str, Any], path: Path):
    labels = ["Full Fine-Tuning", "LoRA (r=8)"]
    params = [full_ft["trainable_params"], lora_r8["trainable_params"]]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, params, color=["tab:blue", "tab:orange"])
    plt.ylabel("Trainable parameters")
    plt.title("Parameter Efficiency")
    plt.yscale("log")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.2, f"{int(yval):,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_extensions_comparison(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    labels = [r.get("variant", r["run_name"]) for r in rows]
    accs = [r["val_accuracy"] for r in rows]
    
    plt.figure(figsize=(8, 4))
    bars = plt.barh(labels, accs, color="tab:green")
    plt.xlabel("Validation accuracy")
    plt.title("LoRA Extensions Comparison")
    plt.xlim(0, 1.0)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.0, f"{width:.4f}", ha="left", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_rank_sweep(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    ranks = [int(row["rank"]) if row["mode"] == "lora" else 0 for row in rows]
    acc = [row["val_accuracy"] for row in rows]
    params = [row["trainable_params"] for row in rows]

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
    if not rows:
        return
    labels = [row.get("combo_name", row["target_modules"]) for row in rows]
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


def main():
    ensure_dirs()
    logs = load_logs()

    fields = [
        "run_name", "mode", "target_modules", "rank", "alpha", 
        "val_accuracy", "trainable_params", "total_params", 
        "trainable_pct"
    ]

    print("[analyze] Processing logs...")

    # --- Baseline ---
    if "baseline_full_ft" in logs and "baseline_lora_r8" in logs:
        baseline_rows = [logs["baseline_full_ft"], logs["baseline_lora_r8"]]
        save_csv(BASELINE_DIR / "baseline.csv", baseline_rows, fields)
        plot_baseline_comparison(logs["baseline_full_ft"], logs["baseline_lora_r8"], BASELINE_DIR / "figures" / "baseline_comparison.png")
        plot_param_efficiency(logs["baseline_full_ft"], logs["baseline_lora_r8"], BASELINE_DIR / "figures" / "param_efficiency.png")

    # --- Rank Sweep ---
    rank_runs = [v for k, v in logs.items() if k.startswith("lora_rank_")]
    if rank_runs and "baseline_full_ft" in logs:
        rank_runs = sorted(rank_runs, key=lambda x: x["rank"])
        rank_rows = [logs["baseline_full_ft"]] + rank_runs
        save_csv(RANK_SWEEP_DIR / "rank_sweep.csv", rank_rows, fields)
        plot_rank_sweep(rank_rows, RANK_SWEEP_DIR / "figures" / "rank_sweep_accuracy.png")

    # --- Module Comparison ---
    module_runs = []
    combo_names = {
        "query": "Wq", "key": "Wk", "value": "Wv", "query,value": "Q+V", "query,key,value": "QKV",
        "attention.output.dense": "Attention_out", "intermediate.dense": "FFN_up", "output.dense": "FFN_down"
    }
    for k, v in logs.items():
        if k.startswith("lora_module_"):
            v["combo_name"] = combo_names.get(v["target_modules"], v["target_modules"])
            module_runs.append(v)
    if module_runs and "baseline_lora_r8" in logs:
        base_lora = dict(logs["baseline_lora_r8"])
        base_lora["combo_name"] = "Q+V"
        module_rows = [base_lora] + module_runs
        save_csv(MODULE_CMP_DIR / "module_comparison.csv", module_rows, fields + ["combo_name"])
        plot_module_comparison(module_rows, MODULE_CMP_DIR / "figures" / "module_comparison_accuracy.png")

    # --- Extensions ---
    ext_keys = ["lora_plus_r8", "lora_dropout_r8", "lora_plus_dropout_r8"]
    ext_runs = []
    for k in ext_keys:
        if k in logs:
            ext_runs.append(logs[k])
    
    if ext_runs and "baseline_lora_r8" in logs:
        base_lora = dict(logs["baseline_lora_r8"])
        base_lora["variant"] = "Baseline LoRA"
        for r in ext_runs:
            if r["run_name"] == "lora_plus_r8": r["variant"] = "LoRA+"
            elif r["run_name"] == "lora_dropout_r8": r["variant"] = "LoRA + Dropout"
            elif r["run_name"] == "lora_plus_dropout_r8": r["variant"] = "LoRA+ + Dropout"
        
        ext_rows = [base_lora] + ext_runs
        save_csv(EXTENSIONS_DIR / "extensions.csv", ext_rows, ["variant"] + fields)
        plot_extensions_comparison(ext_rows, EXTENSIONS_DIR / "figures" / "extensions_comparison.png")

    print("[analyze] All figures and CSVs generated successfully.")


if __name__ == "__main__":
    main()
