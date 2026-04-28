"""
lora.py
-------
Core LoRA implementation following the paper and Microsoft loralib behavior.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Linear):
    """
    LoRA-augmented linear layer.

    By default, this follows Microsoft loralib behavior:
    - A uses Kaiming init and B is zero-initialized.
    - model.eval() merges BA into frozen base weight.
    - model.train() unmerges it back.

    For paper-style init, set `init_method="paper"` (A ~ N(0, 0.02), B = 0).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        init_method: str = "microsoft",
        original_layer: Optional[nn.Linear] = None,
    ):
        bias = original_layer.bias is not None if original_layer is not None else True
        super().__init__(in_features, out_features, bias=bias)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.merged = False
        self.init_method = init_method

        if rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if original_layer is not None:
            self.weight.data.copy_(original_layer.weight.data)
            if self.bias is not None and original_layer.bias is not None:
                self.bias.data.copy_(original_layer.bias.data)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        if self.rank <= 0:
            return
        if self.init_method == "microsoft":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif self.init_method == "paper":
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)
        else:
            raise ValueError(
                f"Unknown LoRA init_method='{self.init_method}'. Use 'microsoft' or 'paper'."
            )

    def _transpose_if_needed(self, w: torch.Tensor) -> torch.Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        super().train(mode)
        if self.rank <= 0:
            return self

        delta_w = self._transpose_if_needed(self.lora_B @ self.lora_A) * self.scaling
        if mode:
            if self.merge_weights and self.merged:
                self.weight.data -= delta_w
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                self.weight.data += delta_w
                self.merged = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self._transpose_if_needed(self.weight), bias=self.bias)
        if self.rank > 0 and not self.merged:
            lora_out = (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            )
            base = base + lora_out * self.scaling
        return base

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"init={self.init_method}, merged={self.merged}"
        )


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """
    Match loralib behavior:
    - bias='none': only LoRA parameters trainable
    - bias='all': all bias parameters trainable too
    - bias='lora_only': only bias inside LoRA-replaced modules trainable
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    if bias == "none":
        return
    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        return
    if bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoraLinear) and module.bias is not None:
                module.bias.requires_grad = True
        return
    raise NotImplementedError(f"Unsupported bias mode '{bias}'")


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    init_method: str = "microsoft",
    merge_weights: bool = True,
    train_classifier: bool = True,
    train_pooler: bool = True,
    train_bias: str = "none",
) -> nn.Module:
    """
    Replace matching nn.Linear layers with LoraLinear in-place.
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name for target in target_modules):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        setattr(
            parent,
            attr,
            LoraLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                merge_weights=merge_weights,
                init_method=init_method,
                original_layer=module,
            ),
        )
        replaced += 1

    mark_only_lora_as_trainable(model, bias=train_bias)

    for name, param in model.named_parameters():
        if train_classifier and "classifier" in name:
            param.requires_grad = True
        if train_pooler and "pooler" in name:
            param.requires_grad = True

    print(
        f"[inject_lora] Replaced {replaced} linear layers -> LoraLinear "
        f"(rank={rank}, alpha={alpha}, init={init_method}, targets={target_modules})"
    )
    _print_param_stats(model)
    return model


def lora_state_dict(
    model: nn.Module,
    bias: str = "none",
    include_classifier: bool = True,
    include_pooler: bool = True,
) -> dict:
    """
    Return LoRA-focused state dict; includes head by default for classifier tasks.
    """
    state = model.state_dict()

    keep_keys = set()
    for key in state:
        if "lora_" in key:
            keep_keys.add(key)

    if bias == "all":
        for key in state:
            if "bias" in key:
                keep_keys.add(key)
    elif bias == "lora_only":
        for key in list(keep_keys):
            bias_key = key.split("lora_")[0] + "bias"
            if bias_key in state:
                keep_keys.add(bias_key)
    elif bias != "none":
        raise NotImplementedError(f"Unsupported bias mode '{bias}'")

    if include_classifier:
        for key in state:
            if "classifier" in key:
                keep_keys.add(key)
    if include_pooler:
        for key in state:
            if "pooler" in key:
                keep_keys.add(key)

    return {k: state[k] for k in state if k in keep_keys}


def _print_param_stats(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[inject_lora] Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
