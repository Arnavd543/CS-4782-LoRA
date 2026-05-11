"""
forgetting.py
-------------
"LoRA learns less, forgets less" — rigorous MLM perplexity test.

For each fine-tuned variant, we evaluate masked-language-modeling perplexity
on a held-out wikitext-2 sample using the pretrained MLM head. The intuition:
fine-tuning on SST-2 drifts the backbone away from pretraining; LoRA's
rank-bounded delta drifts less than full fine-tuning, so its perplexity stays
closer to the pretrained baseline.

Variants:
  - pretrained           (reference)
  - full_ft              (SST-2 full fine-tune)
  - lora_r1 / r8 / r32   (SST-2 LoRA rank sweep)
"""

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


def load_full_ft_backbone(checkpoint_path: Path, mlm_model: RobertaForMaskedLM) -> RobertaForMaskedLM:
    """Copy only `roberta.*` backbone weights from a full-FT classification
    checkpoint into the MLM model. Classifier keys (classifier.*) are ignored
    via strict=False since the MLM model uses lm_head.* instead."""
    state = torch.load(checkpoint_path, map_location="cpu")
    backbone_state = {k: v for k, v in state.items() if k.startswith("roberta.")}
    missing, unexpected = mlm_model.load_state_dict(backbone_state, strict=False)
    print(f"  [full_ft] loaded {len(backbone_state)} backbone keys "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    return mlm_model


def load_lora_backbone(
    checkpoint_path: Path,
    mlm_model: RobertaForMaskedLM,
    rank: int,
    alpha: float = 8.0,
    target_modules: Optional[List[str]] = None,
) -> RobertaForMaskedLM:
    """Inject LoRA into the MLM model's attention query/value layers, then load
    the saved lora_A/lora_B weights from a LoRA checkpoint."""
    if target_modules is None:
        target_modules = ["query", "value"]
    inject_lora(
        mlm_model,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        init_method="paper",
        merge_weights=True,
        train_classifier=False,
        train_pooler=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    lora_state = {k: v for k, v in state.items() if "lora_" in k}
    missing, unexpected = mlm_model.load_state_dict(lora_state, strict=False)
    print(f"  [lora r={rank}] loaded {len(lora_state)} LoRA keys "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    return mlm_model


@torch.no_grad()
def compute_perplexity(
    model: RobertaForMaskedLM,
    tokenizer,
    texts: List[str],
    device: torch.device,
    mask_prob: float = 0.15,
    max_length: int = 256,
    seed: int = 0,
) -> float:
    """Standard masked-LM perplexity. We mask `mask_prob` of tokens, set the
    rest to -100 so they don't contribute to the loss, and sum loss over masked
    tokens only. Perplexity = exp(loss / num_masked)."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)

    total_loss = 0.0
    total_masked = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        # Mask `mask_prob` of non-special tokens
        special_tokens_mask = tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device).unsqueeze(0)

        prob_matrix = torch.full(labels.shape, mask_prob, device="cpu")
        masked_indices_cpu = torch.bernoulli(prob_matrix, generator=g).bool()
        masked_indices = masked_indices_cpu.to(device) & ~special_tokens_mask & attention_mask.bool()

        if masked_indices.sum() == 0:
            continue

        labels[~masked_indices] = -100
        masked_input_ids = input_ids.clone()
        masked_input_ids[masked_indices] = tokenizer.mask_token_id

        outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels)
        n_masked = int(masked_indices.sum().item())
        total_loss += float(outputs.loss.item()) * n_masked
        total_masked += n_masked

    if total_masked == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_loss / total_masked)).item())


def build_variant(
    variant: str,
    base_model_name: str,
    device: torch.device,
) -> RobertaForMaskedLM:
    """Construct an MLM model for the given variant."""
    model = RobertaForMaskedLM.from_pretrained(base_model_name)

    if variant == "pretrained":
        pass  # use as-is
    elif variant == "full_ft":
        ckpt = CHECKPOINTS_DIR / "baseline_full_ft_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model = load_full_ft_backbone(ckpt, model)
    elif variant.startswith("lora_r"):
        rank = int(variant.split("_r")[1])
        # The notebook saves rank-sweep checkpoints as lora_rank_{r}_best.pt
        ckpt = CHECKPOINTS_DIR / f"lora_rank_{rank}_best.pt"
        if not ckpt.exists():
            # Fall back to baseline LoRA r=8 if rank=8 sweep checkpoint missing
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
    """Bar chart with pretrained as the leftmost (reference) bar."""
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
    # Reference line at pretrained perplexity
    pre_ppl = rows[0]["perplexity"]
    plt.axhline(pre_ppl, color="tab:gray", linestyle="--", alpha=0.5, label="pretrained")
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
    print(f"[forgetting] Device: {device}")
    print(f"[forgetting] Loading wikitext-2 test split...")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_texts = [t for t in ds["text"] if len(t.strip()) > 50]
    texts = raw_texts[: args.num_texts]
    print(f"[forgetting] Evaluating on {len(texts)} texts.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows: List[Dict] = []
    pre_ppl = None

    for variant in variants:
        print(f"\n[forgetting] === {variant} ===")
        try:
            model = build_variant(variant, args.base_model, device)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        ppl = compute_perplexity(
            model,
            tokenizer,
            texts,
            device,
            mask_prob=args.mask_prob,
            max_length=args.max_length,
            seed=args.seed,
        )
        if pre_ppl is None and variant == "pretrained":
            pre_ppl = ppl

        increase = (ppl - pre_ppl) if pre_ppl is not None else 0.0
        pct = (100.0 * increase / pre_ppl) if pre_ppl else 0.0

        print(f"  perplexity = {ppl:.4f}")
        if pre_ppl is not None:
            print(f"  delta vs pretrained = +{increase:.4f} ({pct:+.2f}%)")

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

    print(f"\n[forgetting] Saved {FORGETTING_DIR / 'perplexity.csv'}")
    print(f"[forgetting] Saved {FIGURES_DIR / 'perplexity_comparison.png'}")


if __name__ == "__main__":
    main()
