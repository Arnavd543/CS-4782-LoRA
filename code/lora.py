import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Linear):
    # LoRA-augmented linear layer (paper init: A~N(0,0.02), B=0; eval merges BA into frozen weight).

    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        fanInFanOut: bool = False,
        mergeWeights: bool = True,
        initMethod: str = "paper",
        originalLayer: Optional[nn.Linear] = None,
    ):
        bias = originalLayer.bias is not None if originalLayer is not None else True
        super().__init__(inFeatures, outFeatures, bias=bias)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.fanInFanOut = fanInFanOut
        self.mergeWeights = mergeWeights
        self.merged = False
        self.initMethod = initMethod

        if rank > 0:
            self.loraA = nn.Parameter(self.weight.new_zeros((rank, inFeatures)))
            self.loraB = nn.Parameter(self.weight.new_zeros((outFeatures, rank)))
            self.weight.requires_grad = False
        else:
            self.register_parameter("loraA", None)
            self.register_parameter("loraB", None)

        if originalLayer is not None:
            self.weight.data.copy_(originalLayer.weight.data)
            if self.bias is not None and originalLayer.bias is not None:
                self.bias.data.copy_(originalLayer.bias.data)

        self.loraDropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if self.fanInFanOut:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        if self.rank <= 0:
            return
        if self.initMethod == "microsoft":
            nn.init.kaiming_uniform_(self.loraA, a=math.sqrt(5))
            nn.init.zeros_(self.loraB)
        elif self.initMethod == "paper":
            nn.init.normal_(self.loraA, mean=0.0, std=0.02)
            nn.init.zeros_(self.loraB)
        else:
            raise ValueError(
                f"Unknown LoRA initMethod='{self.initMethod}'. Use 'microsoft' or 'paper'."
            )

    def _transpose_if_needed(self, w: torch.Tensor) -> torch.Tensor:
        return w.transpose(0, 1) if self.fanInFanOut else w

    def train(self, mode: bool = True):
        super().train(mode)
        if self.rank <= 0:
            return self

        deltaW = self._transpose_if_needed(self.loraB @ self.loraA) * self.scaling
        if mode:
            if self.mergeWeights and self.merged:
                self.weight.data -= deltaW
                self.merged = False
        else:
            if self.mergeWeights and not self.merged:
                self.weight.data += deltaW
                self.merged = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self._transpose_if_needed(self.weight), bias=self.bias)
        if self.rank > 0 and not self.merged:
            loraOut = (
                self.loraDropout(x)
                @ self.loraA.transpose(0, 1)
                @ self.loraB.transpose(0, 1)
            )
            base = base + loraOut * self.scaling
        return base

    def extra_repr(self) -> str:
        return (
            "in_features=" + str(self.in_features) + ", out_features=" + str(self.out_features) +
            ", rank=" + str(self.rank) + ", alpha=" + str(self.alpha) +
            ", scaling=" + format(self.scaling, ".4f") +
            ", init=" + self.initMethod + ", merged=" + str(self.merged)
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
    targetModules: Optional[List[str]] = None,
    initMethod: str = "paper",
    mergeWeights: bool = True,
    trainClassifier: bool = True,
    trainPooler: bool = True,
    trainBias: str = "none",
) -> nn.Module:
    # Replace matching nn.Linear layers with LoraLinear in-place.
    if targetModules is None:
        targetModules = ["query", "value"]

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name for target in targetModules):
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
                inFeatures=module.in_features,
                outFeatures=module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                mergeWeights=mergeWeights,
                initMethod=initMethod,
                originalLayer=module,
            ),
        )
        replaced += 1

    mark_only_lora_as_trainable(model, bias=trainBias)

    for name, param in model.named_parameters():
        if trainClassifier and "classifier" in name:
            param.requires_grad = True
        if trainPooler and "pooler" in name:
            param.requires_grad = True

    print(
        "[inject_lora] Replaced " + str(replaced) + " linear layers -> LoraLinear "
        "(rank=" + str(rank) + ", alpha=" + str(alpha) + ", init=" + initMethod + ", targets=" + str(targetModules) + ")"
    )
    _print_param_stats(model)
    return model


def lora_state_dict(
    model: nn.Module,
    bias: str = "none",
    includeClassifier: bool = True,
    includePooler: bool = True,
) -> dict:
    # Return LoRA-focused state dict; includes classifier head by default.
    state = model.state_dict()

    keepKeys = set()
    for key in state:
        if "loraA" in key or "loraB" in key:
            keepKeys.add(key)

    if bias == "all":
        for key in state:
            if "bias" in key:
                keepKeys.add(key)
    elif bias == "lora_only":
        for key in list(keepKeys):
            if "loraA" in key:
                prefix = key[:key.rfind(".loraA")]
            elif "loraB" in key:
                prefix = key[:key.rfind(".loraB")]
            else:
                continue
            biasKey = prefix + ".bias"
            if biasKey in state:
                keepKeys.add(biasKey)
    elif bias != "none":
        raise NotImplementedError(f"Unsupported bias mode '{bias}'")

    if includeClassifier:
        for key in state:
            if "classifier" in key:
                keepKeys.add(key)
    if includePooler:
        for key in state:
            if "pooler" in key:
                keepKeys.add(key)

    return {k: state[k] for k in state if k in keepKeys}


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
