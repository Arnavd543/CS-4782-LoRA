import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Linear):
    # LoRA-augmented linear layer (paper init: A~N(0,0.02), B=0; eval merges BA into frozen weight).

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        merge_weights: bool = True,
        init_method: str = "paper",
        original_layer: Optional[nn.Linear] = None,
    ):
        bias = original_layer.bias is not None if original_layer is not None else True
        super().__init__(in_features, out_features, bias=bias)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.merge_weights = merge_weights
        self.merged = False
        self.init_method = init_method

        # loraA / loraB attribute names are kept as-is (paper notation A, B) so that existing
        # checkpoint keys (e.g. "...query.loraA") continue to load.
        if rank > 0:
            self.loraA = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.loraB = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("loraA", None)
            self.register_parameter("loraB", None)

        if original_layer is not None:
            self.weight.data.copy_(original_layer.weight.data)
            if self.bias is not None and original_layer.bias is not None:
                self.bias.data.copy_(original_layer.bias.data)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        if self.rank <= 0:
            return
        if self.init_method == "microsoft":
            nn.init.kaiming_uniform_(self.loraA, a=math.sqrt(5))
            nn.init.zeros_(self.loraB)
        elif self.init_method == "paper":
            nn.init.normal_(self.loraA, mean=0.0, std=0.02)
            nn.init.zeros_(self.loraB)
        else:
            raise ValueError(
                f"Unknown LoRA init_method='{self.init_method}'. Use 'microsoft' or 'paper'."
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.rank <= 0:
            return self

        delta_w = (self.loraB @ self.loraA) * self.scaling
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
        base = F.linear(x, self.weight, bias=self.bias)
        if self.rank > 0 and not self.merged:
            lora_out = (
                self.lora_dropout(x)
                @ self.loraA.transpose(0, 1)
                @ self.loraB.transpose(0, 1)
            )
            base = base + lora_out * self.scaling
        return base

    def extra_repr(self) -> str:
        return (
            "in_features=" + str(self.in_features) + ", out_features=" + str(self.out_features) +
            ", rank=" + str(self.rank) + ", alpha=" + str(self.alpha) +
            ", scaling=" + format(self.scaling, ".4f") +
            ", init=" + self.init_method + ", merged=" + str(self.merged)
        )


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    # Freeze all params except loraA/loraB (and optionally biases).
    for name, param in model.named_parameters():
        if "loraA" not in name and "loraB" not in name:
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
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    init_method: str = "paper",
    merge_weights: bool = True,
    train_classifier: bool = True,
    train_pooler: bool = True,
    train_bias: str = "none",
) -> nn.Module:
    # Replace matching nn.Linear layers with LoraLinear in-place.
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
        "[inject_lora] Replaced " + str(replaced) + " linear layers -> LoraLinear "
        "(rank=" + str(rank) + ", alpha=" + str(alpha) + ", init=" + init_method + ", targets=" + str(target_modules) + ")"
    )
    _print_param_stats(model)
    return model


def lora_state_dict(
    model: nn.Module,
    bias: str = "none",
    include_classifier: bool = True,
    include_pooler: bool = True,
) -> dict:
    # Return LoRA-focused state dict; includes classifier head by default.
    state = model.state_dict()

    keep_keys = set()
    for key in state:
        if "loraA" in key or "loraB" in key:
            keep_keys.add(key)

    if bias == "all":
        for key in state:
            if "bias" in key:
                keep_keys.add(key)
    elif bias == "lora_only":
        for key in list(keep_keys):
            if "loraA" in key:
                prefix = key[: key.rfind(".loraA")]
            elif "loraB" in key:
                prefix = key[: key.rfind(".loraB")]
            else:
                continue
            bias_key = prefix + ".bias"
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
        "[inject_lora] Trainable: " + format(trainable, ",") + " / " + format(total, ",") +
        " (" + format(100 * trainable / total, ".4f") + "%)"
    )


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
