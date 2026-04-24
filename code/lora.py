"""
lora.py
-------
Core LoRA implementation following Hu et al. 2021 (arXiv:2106.09685).

Key components:
    LoraLinear  – drop-in replacement for nn.Linear with low-rank adapters.
    inject_lora – walks a model and replaces target modules in-place.
    lora_state_dict – returns only the trainable LoRA + head parameters.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class LoraLinear(nn.Module):
    """
    A linear layer augmented with a low-rank adapter.

    Forward pass:
        h = W₀ x  +  (x Aᵀ Bᵀ) * (alpha / rank)

    W₀ is frozen. A and B are the only trainable parameters in this layer.

    Args:
        in_features:  input dimension of the original linear layer.
        out_features: output dimension of the original linear layer.
        rank:         rank r of the low-rank decomposition.
        alpha:        scaling factor alpha (paper recommends alpha = 2 * rank).
        dropout:      dropout probability applied between A and B (0 = disabled).
        original_layer: the nn.Linear being replaced; its weight is copied and frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        original_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # ── Frozen pretrained weight ──────────────────────────────────────────
        self.weight = nn.Parameter(
            original_layer.weight.data.clone() if original_layer is not None
            else torch.empty(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None
        if original_layer is not None and original_layer.bias is not None:
            self.bias = nn.Parameter(
                original_layer.bias.data.clone(), requires_grad=False
            )

        # ── Trainable low-rank matrices ───────────────────────────────────────
        # A: shape (rank, in_features)  — Gaussian init (paper §4)
        # B: shape (out_features, rank) — zero init so ΔW = BA = 0 at step 0
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

        # Optional dropout between A and B (Improvement 3)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base computation
        base = nn.functional.linear(x, self.weight, self.bias)

        # Low-rank adapter: x → A → dropout → B → scale
        # x @ Aᵀ  : (..., in_features) → (..., rank)
        # result @ Bᵀ : (..., rank) → (..., out_features)
        lora_out = self.lora_dropout(x @ self.lora_A.t()) @ self.lora_B.t()

        return base + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


# ── Injection ──────────────────────────────────────────────────────────────────

def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers whose name matches *any* entry in
    `target_modules` with LoraLinear, preserving pretrained weights.

    Args:
        model:          the pretrained model to modify (mutated in-place).
        rank:           LoRA rank r.
        alpha:          LoRA scaling alpha.
        dropout:        dropout probability inside the LoRA path.
        target_modules: list of substring matches, e.g. ["query", "value"].
                        If None, defaults to ["query", "value"] (paper default).

    Returns:
        model with LoRA layers injected and all non-LoRA params frozen.
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # Navigate to the parent module so we can do setattr
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        lora_layer = LoraLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            original_layer=module,
        )
        setattr(parent, attr, lora_layer)
        replaced += 1

    print(f"[inject_lora] Replaced {replaced} linear layers → LoraLinear "
          f"(rank={rank}, alpha={alpha}, targets={target_modules})")

    # Unfreeze classification head (anything named 'classifier' or 'pooler')
    for name, param in model.named_parameters():
        if any(k in name for k in ["classifier", "pooler", "lora_"]):
            param.requires_grad = True

    _print_param_stats(model)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Return only trainable parameters — safe to save as a small checkpoint."""
    return {k: v for k, v in model.state_dict().items()
            if any(n in k for n in ["lora_A", "lora_B", "classifier", "pooler"])}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_param_stats(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[inject_lora] Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.4f}%)")


def count_parameters(model: nn.Module):
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ── Improvement 1: LoRA+ — asymmetric learning rates ──────────────────────────

def lora_plus_param_groups(model: nn.Module, lr_A: float, lr_B_multiplier: float = 16.0):
    """
    Return optimizer param groups with separate (lower) lr for lora_A
    and (higher) lr for lora_B, following LoRA+ (Hayou et al. 2024).

    Usage:
        groups = lora_plus_param_groups(model, lr_A=1e-4, lr_B_multiplier=16.0)
        optimizer = torch.optim.AdamW(groups)
    """
    lora_a, lora_b, other = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a.append(param)
        elif "lora_B" in name:
            lora_b.append(param)
        else:
            other.append(param)

    return [
        {"params": lora_a, "lr": lr_A},
        {"params": lora_b, "lr": lr_A * lr_B_multiplier},
        {"params": other,  "lr": lr_A},
    ]
