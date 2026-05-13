import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def ensure_dirs(dirs: Dict[str, Path]):
    top_figures = dirs["top_figures"]
    for key, path in dirs.items():
        if key in ("top_figures", "logs"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.mkdir(parents=True, exist_ok=True)
        if path != top_figures:
            (path / "figures").mkdir(exist_ok=True)


def load_logs(logs_dir: Path) -> Dict[str, Dict[str, Any]]:
    logs = {}
    if not logs_dir.exists():
        print("No logs directory found at " + str(logs_dir))
        return logs
    for p in logs_dir.glob("*.json"):
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
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val + 0.01, f"{y_val:.4f}", ha="center", va="bottom")
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
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val * 1.2, f"{int(y_val):,}", ha="center", va="bottom")
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
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2.0, f"{width:.4f}", ha="left", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_rank_sweep(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    ranks = [int(row["rank"]) if row["mode"] == "lora" else 0 for row in rows]
    full_acc = rows[0]["val_accuracy"]
    full_params = rows[0]["trainable_params"]

    plt.figure(figsize=(8, 5))
    plt.plot(
        [0] + [r for r in ranks if r != 0],
        [full_acc] + [row["val_accuracy"] for row in rows if row["mode"] == "lora"],
        marker="o",
    )
    plt.xticks([0] + [r for r in ranks if r != 0], ["full"] + [str(r) for r in ranks if r != 0])
    plt.xlabel("LoRA rank")
    plt.ylabel("Validation accuracy")
    plt.title("LoRA rank sweep vs full fine-tuning")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        [0] + [r for r in ranks if r != 0],
        [full_params] + [row["trainable_params"] for row in rows if row["mode"] == "lora"],
        marker="o",
    )
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


def _extract_epoch_accuracy_series(log_history: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    epochs, accs = [], []
    for entry in log_history or []:
        if "eval_accuracy" in entry and "epoch" in entry:
            epochs.append(entry["epoch"])
            accs.append(entry["eval_accuracy"])
    return epochs, accs


def plot_convergence_curves(runs: List[Dict[str, Any]], labels: List[str], path: Path, title: str):
    if not runs:
        return
    plt.figure(figsize=(8, 5))
    plotted = 0
    for run, label in zip(runs, labels):
        epochs, accs = _extract_epoch_accuracy_series(run.get("log_history", []))
        if not epochs:
            continue
        plt.plot(epochs, accs, marker="o", label=label)
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_time_comparison(rows: List[Dict[str, Any]], path: Path, title: str = "Training time"):
    if not rows:
        return
    labels = [r.get("variant", r.get("combo_name", r["run_name"])) for r in rows]
    times = [r.get("elapsed_sec", 0.0) / 60.0 for r in rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, times, color="tab:purple")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Training time (minutes)")
    plt.title(title)
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val + 0.1, f"{y_val:.1f}m", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_memory_comparison(rows: List[Dict[str, Any]], path: Path, title: str = "Peak GPU memory"):
    if not rows:
        return
    labels = [r.get("variant", r.get("combo_name", r["run_name"])) for r in rows]
    mem = [r.get("gpu_memory_mb", 0.0) for r in rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, mem, color="tab:red")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title(title)
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val + 50, f"{int(y_val)}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_time_vs_accuracy(all_rows: List[Tuple[str, Dict[str, Any]]], path: Path):
    if not all_rows:
        return
    families = sorted({family for family, _ in all_rows})
    colors = {f: c for f, c in zip(families, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])}

    plt.figure(figsize=(9, 6))
    for family in families:
        xs = [row["elapsed_sec"] / 60.0 for f, row in all_rows if f == family and row.get("elapsed_sec")]
        ys = [row["val_accuracy"] for f, row in all_rows if f == family and row.get("elapsed_sec")]
        plt.scatter(xs, ys, label=family, color=colors[family], s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    plt.xlabel("Training time (minutes)")
    plt.ylabel("Validation accuracy")
    plt.title("Time vs accuracy across all experiments")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_param_efficiency_scatter(all_rows: List[Tuple[str, Dict[str, Any]]], path: Path):
    if not all_rows:
        return
    families = sorted({family for family, _ in all_rows})
    colors = {f: c for f, c in zip(families, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])}

    plt.figure(figsize=(9, 6))
    for family in families:
        xs = [row["trainable_params"] for f, row in all_rows if f == family]
        ys = [row["val_accuracy"] for f, row in all_rows if f == family]
        plt.scatter(xs, ys, label=family, color=colors[family], s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    plt.xscale("log")
    plt.xlabel("Trainable parameters (log scale)")
    plt.ylabel("Validation accuracy")
    plt.title("Parameter efficiency: accuracy vs trainable params")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_combined_accuracy(all_rows: List[Tuple[str, Dict[str, Any]]], path: Path):
    if not all_rows:
        return
    families = sorted({family for family, _ in all_rows})
    colors = {f: c for f, c in zip(families, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])}

    grouped = []
    for fam in families:
        fam_rows = [(f, r) for f, r in all_rows if f == fam]
        fam_rows.sort(key=lambda x: x[1]["val_accuracy"], reverse=True)
        grouped.extend(fam_rows)

    labels = ["[" + fam + "] " + row["run_name"] for fam, row in grouped]
    accs = [row["val_accuracy"] for _, row in grouped]
    bar_colors = [colors[fam] for fam, _ in grouped]

    plt.figure(figsize=(10, max(5, len(labels) * 0.3)))
    bars = plt.barh(labels, accs, color=bar_colors)
    plt.xlabel("Validation accuracy")
    plt.title("All experiments — combined accuracy comparison")
    plt.xlim(0, 1.0)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2.0, f"{width:.3f}", ha="left", va="center", fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-run JSON logs into CSVs and figures.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Root results directory containing logs/ and per-experiment subdirs.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dirs = {
        "results": results_dir,
        "logs": results_dir / "logs",
        "top_figures": results_dir / "figures",
        "baseline": results_dir / "baseline",
        "rank_sweep": results_dir / "rank_sweep",
        "module_comparison": results_dir / "module_comparison",
        "extensions": results_dir / "extensions",
        "mrpc": results_dir / "mrpc",
    }
    logs_dir = dirs["logs"]
    top_figures_dir = dirs["top_figures"]
    baseline_dir = dirs["baseline"]
    rank_sweep_dir = dirs["rank_sweep"]
    module_cmp_dir = dirs["module_comparison"]
    extensions_dir = dirs["extensions"]
    mrpc_dir = dirs["mrpc"]

    ensure_dirs(dirs)
    logs = load_logs(logs_dir)

    fields = [
        "run_name", "task", "mode", "target_modules", "rank", "alpha",
        "val_accuracy", "trainable_params", "total_params",
        "trainable_pct", "elapsed_sec", "gpu_memory_mb",
    ]

    print("[analyze] Processing logs...")
    baseline_lora_key = "baseline_lora_r8_paper"

    sst2_logs = {k: v for k, v in logs.items() if v.get("task", "sst2") == "sst2"}

    if "baseline_full_ft" in sst2_logs and baseline_lora_key in sst2_logs:
        baseline_rows = [sst2_logs["baseline_full_ft"], sst2_logs[baseline_lora_key]]
        save_csv(baseline_dir / "baseline.csv", baseline_rows, fields)
        plot_baseline_comparison(
            sst2_logs["baseline_full_ft"],
            sst2_logs[baseline_lora_key],
            baseline_dir / "figures" / "baseline_comparison.png",
        )
        plot_param_efficiency(
            sst2_logs["baseline_full_ft"],
            sst2_logs[baseline_lora_key],
            baseline_dir / "figures" / "param_efficiency.png",
        )
        plot_convergence_curves(
            [sst2_logs["baseline_full_ft"], sst2_logs[baseline_lora_key]],
            ["Full FT", "LoRA r=8"],
            baseline_dir / "figures" / "convergence_curves.png",
            "Baseline convergence (SST-2)",
        )
        plot_memory_comparison(
            [
                {**sst2_logs["baseline_full_ft"], "variant": "Full FT"},
                {**sst2_logs[baseline_lora_key], "variant": "LoRA r=8"},
            ],
            baseline_dir / "figures" / "memory_comparison.png",
            "Peak GPU memory: Full FT vs LoRA r=8",
        )

    rank_runs = sorted(
        [v for k, v in sst2_logs.items() if k.startswith("lora_rank_")],
        key=lambda x: x["rank"],
    )
    if rank_runs and "baseline_full_ft" in sst2_logs:
        rank_rows = [sst2_logs["baseline_full_ft"]] + rank_runs
        save_csv(rank_sweep_dir / "rank_sweep.csv", rank_rows, fields)
        plot_rank_sweep(rank_rows, rank_sweep_dir / "figures" / "rank_sweep_accuracy.png")

    combo_names = {
        "query": "Wq", "key": "Wk", "value": "Wv",
        "query,value": "Q+V", "query,key,value": "QKV",
        "attention.output.dense": "Attention_out",
        "intermediate.dense": "FFN_up", "output.dense": "FFN_down",
    }
    module_runs = []
    for k, v in sst2_logs.items():
        if k.startswith("lora_module_"):
            v["combo_name"] = combo_names.get(v["target_modules"], v["target_modules"])
            module_runs.append(v)
    if module_runs and baseline_lora_key in sst2_logs:
        base_lora = dict(sst2_logs[baseline_lora_key])
        base_lora["combo_name"] = "Q+V"
        module_rows = [base_lora] + module_runs
        save_csv(module_cmp_dir / "module_comparison.csv", module_rows, fields + ["combo_name"])
        plot_module_comparison(module_rows, module_cmp_dir / "figures" / "module_comparison_accuracy.png")
        plot_time_comparison(
            module_rows,
            module_cmp_dir / "figures" / "time_comparison.png",
            "Training time by target module",
        )

    ext_keys = ["lora_plus_r8", "lora_dropout_r8", "lora_plus_dropout_r8"]
    ext_runs = []
    for k in ext_keys:
        if k in sst2_logs:
            ext_runs.append(sst2_logs[k])

    if ext_runs and baseline_lora_key in sst2_logs:
        base_lora = dict(sst2_logs[baseline_lora_key])
        base_lora["variant"] = "Baseline LoRA"
        for r in ext_runs:
            if r["run_name"] == "lora_plus_r8":
                r["variant"] = "LoRA+"
            elif r["run_name"] == "lora_dropout_r8":
                r["variant"] = "LoRA + Dropout"
            elif r["run_name"] == "lora_plus_dropout_r8":
                r["variant"] = "LoRA+ + Dropout"

        ext_rows = [base_lora] + ext_runs
        save_csv(extensions_dir / "extensions.csv", ext_rows, ["variant"] + fields)
        plot_extensions_comparison(ext_rows, extensions_dir / "figures" / "extensions_comparison.png")
        plot_convergence_curves(
            ext_rows,
            [r["variant"] for r in ext_rows],
            extensions_dir / "figures" / "convergence_curves.png",
            "Extensions convergence (SST-2)",
        )
        plot_time_comparison(
            ext_rows,
            extensions_dir / "figures" / "time_comparison.png",
            "Training time by extension variant",
        )

    mrpc_logs = {k: v for k, v in logs.items() if v.get("task") == "mrpc"}
    if "mrpc_full_ft" in mrpc_logs and "mrpc_lora_r8" in mrpc_logs:
        mrpc_rows = [mrpc_logs["mrpc_full_ft"], mrpc_logs["mrpc_lora_r8"]]
        save_csv(mrpc_dir / "baseline.csv", mrpc_rows, fields)
        plot_baseline_comparison(
            mrpc_logs["mrpc_full_ft"],
            mrpc_logs["mrpc_lora_r8"],
            mrpc_dir / "figures" / "baseline_comparison.png",
        )
        plot_param_efficiency(
            mrpc_logs["mrpc_full_ft"],
            mrpc_logs["mrpc_lora_r8"],
            mrpc_dir / "figures" / "param_efficiency.png",
        )

    all_rows: List[Tuple[str, Dict[str, Any]]] = []
    if "baseline_full_ft" in sst2_logs:
        all_rows.append(("baseline", sst2_logs["baseline_full_ft"]))
    if baseline_lora_key in sst2_logs:
        all_rows.append(("baseline", sst2_logs[baseline_lora_key]))
    for run in rank_runs:
        all_rows.append(("rank_sweep", run))
    for run in module_runs:
        all_rows.append(("module_comparison", run))
    for run in ext_runs:
        all_rows.append(("extensions", run))
    for k in ("mrpc_full_ft", "mrpc_lora_r8"):
        if k in mrpc_logs:
            all_rows.append(("mrpc", mrpc_logs[k]))

    if all_rows:
        plot_time_vs_accuracy(all_rows, top_figures_dir / "time_vs_accuracy.png")
        plot_param_efficiency_scatter(all_rows, top_figures_dir / "param_efficiency_scatter.png")
        plot_combined_accuracy(all_rows, top_figures_dir / "combined_accuracy.png")

    print("[analyze] All figures and CSVs generated successfully.")


if __name__ == "__main__":
    main()
