# MLM perplexity drift after SST-2 fine-tuning (LoRA vs full FT).

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForMaskedLM

sys.path.insert(0, os.path.dirname(__file__))
from lora import inject_lora


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
FORGETTING_DIR = RESULTS_DIR / "forgetting"
FIGURES_DIR = FORGETTING_DIR / "figures"


def load_full_ft_backbone(checkpointPath: Path, mlmModel: RobertaForMaskedLM) -> RobertaForMaskedLM:
    state = torch.load(checkpointPath, map_location="cpu")
    backboneState = {k: v for k, v in state.items() if k.startswith("roberta.")}
    missing, unexpected = mlmModel.load_state_dict(backboneState, strict=False)
    print("  [full_ft] loaded " + str(len(backboneState)) + " backbone keys "
          "(missing=" + str(len(missing)) + " unexpected=" + str(len(unexpected)) + ")")
    return mlmModel


def load_lora_backbone(
    checkpointPath: Path,
    mlmModel: RobertaForMaskedLM,
    rank: int,
    alpha: float = 8.0,
    targetModules: Optional[List[str]] = None,
) -> RobertaForMaskedLM:
    if targetModules is None:
        targetModules = ["query", "value"]
    inject_lora(
        mlmModel,
        rank=rank,
        alpha=alpha,
        targetModules=targetModules,
        initMethod="paper",
        mergeWeights=True,
        trainClassifier=False,
        trainPooler=False,
    )
    state = torch.load(checkpointPath, map_location="cpu")
    loraState = {k: v for k, v in state.items() if "loraA" in k or "loraB" in k}
    missing, unexpected = mlmModel.load_state_dict(loraState, strict=False)
    print("  [lora r=" + str(rank) + "] loaded " + str(len(loraState)) + " LoRA keys "
          "(missing=" + str(len(missing)) + " unexpected=" + str(len(unexpected)) + ")")
    return mlmModel


@torch.no_grad()
def compute_perplexity(
    model: RobertaForMaskedLM,
    tokenizer,
    texts: List[str],
    device: torch.device,
    maskProb: float = 0.15,
    maxLength: int = 256,
    seed: int = 0,
) -> float:
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)

    totalLoss = 0.0
    totalMasked = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=maxLength,
        )
        inputIds = inputs["input_ids"].to(device)
        attentionMask = inputs["attention_mask"].to(device)
        labels = inputIds.clone()

        specialTokensMask = tokenizer.get_special_tokens_mask(
            inputIds[0].tolist(), already_has_special_tokens=True
        )
        specialTokensMask = torch.tensor(specialTokensMask, dtype=torch.bool, device=device).unsqueeze(0)

        probMatrix = torch.full(labels.shape, maskProb, device="cpu")
        maskedIndicesCpu = torch.bernoulli(probMatrix, generator=g).bool()
        maskedIndices = maskedIndicesCpu.to(device) & ~specialTokensMask & attentionMask.bool()

        if maskedIndices.sum() == 0:
            continue

        labels[~maskedIndices] = -100
        maskedInputIds = inputIds.clone()
        maskedInputIds[maskedIndices] = tokenizer.mask_token_id

        outputs = model(input_ids=maskedInputIds, attention_mask=attentionMask, labels=labels)
        nMasked = int(maskedIndices.sum().item())
        totalLoss += float(outputs.loss.item()) * nMasked
        totalMasked += nMasked

    if totalMasked == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(totalLoss / totalMasked)).item())


def build_variant(
    variant: str,
    baseModelName: str,
    device: torch.device,
) -> RobertaForMaskedLM:
    model = RobertaForMaskedLM.from_pretrained(baseModelName)

    if variant == "pretrained":
        pass
    elif variant == "full_ft":
        ckpt = CHECKPOINTS_DIR / "baseline_full_ft_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model = load_full_ft_backbone(ckpt, model)
    elif variant.startswith("lora_r"):
        rank = int(variant.split("_r")[1])
        ckpt = CHECKPOINTS_DIR / ("lora_rank_" + str(rank) + "_best.pt")
        if not ckpt.exists():
            if rank == 8:
                alt = CHECKPOINTS_DIR / "baseline_lora_r8_paper_best.pt"
                if alt.exists():
                    ckpt = alt
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing LoRA r={rank} checkpoint: {ckpt}")
        model = load_lora_backbone(ckpt, model, rank=rank, alpha=rank)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model.to(device)


def save_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_perplexity(rows: List[Dict], path: Path):
    if not rows:
        return
    labels = [r["variant"] for r in rows]
    ppls = [r["perplexity"] for r in rows]

    plt.figure(figsize=(8, 5))
    colors = ["tab:gray"] + ["tab:blue" if "full" in v else "tab:orange" for v in labels[1:]]
    bars = plt.bar(labels, ppls, color=colors)
    plt.ylabel("Masked-LM perplexity (lower = closer to pretrained)")
    plt.title("LoRA forgets less: perplexity drift after SST-2 fine-tuning")
    for bar, p in zip(bars, ppls):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{p:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    prePpl = rows[0]["perplexity"]
    plt.axhline(prePpl, color="tab:gray", linestyle="--", alpha=0.5, label="pretrained")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MLM perplexity comparison: full FT vs LoRA at multiple ranks.")
    parser.add_argument("--num_texts", type=int, default=500, help="Number of wikitext samples to evaluate on.")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="MLM masking probability.")
    parser.add_argument("--max_length", type=int, default=256, help="Max tokenization length.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for token masking.")
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument(
        "--variants",
        type=str,
        default="pretrained,full_ft,lora_r1,lora_r8,lora_r32",
        help="Comma-separated variant names.",
    )
    args = parser.parse_args()

    FORGETTING_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[forgetting] Device: " + str(device))
    print("[forgetting] Loading wikitext-2 test split...")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    rawTexts = [t for t in ds["text"] if len(t.strip()) > 50]
    texts = rawTexts[: args.num_texts]
    print("[forgetting] Evaluating on " + str(len(texts)) + " texts.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows: List[Dict] = []
    prePpl = None

    for variant in variants:
        print("\n[forgetting] === " + variant + " ===")
        try:
            model = build_variant(variant, args.base_model, device)
        except FileNotFoundError as e:
            print("  SKIPPED: " + str(e))
            continue

        ppl = compute_perplexity(
            model,
            tokenizer,
            texts,
            device,
            maskProb=args.mask_prob,
            maxLength=args.max_length,
            seed=args.seed,
        )
        if prePpl is None and variant == "pretrained":
            prePpl = ppl

        increase = (ppl - prePpl) if prePpl is not None else 0.0
        pct = (100.0 * increase / prePpl) if prePpl else 0.0

        print("  perplexity = " + format(ppl, ".4f"))
        if prePpl is not None:
            print("  delta vs pretrained = +" + format(increase, ".4f") + " (" + format(pct, "+.2f") + "%)")

        rows.append({
            "variant": variant,
            "perplexity": round(ppl, 4),
            "perplexity_increase_vs_pretrained": round(increase, 4),
            "pct_increase_vs_pretrained": round(pct, 2),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_csv(FORGETTING_DIR / "perplexity.csv", rows)
    plot_perplexity(rows, FIGURES_DIR / "perplexity_comparison.png")

    print("\n[forgetting] Saved " + str(FORGETTING_DIR / "perplexity.csv"))
    print("[forgetting] Saved " + str(FIGURES_DIR / "perplexity_comparison.png"))


if __name__ == "__main__":
    main()
