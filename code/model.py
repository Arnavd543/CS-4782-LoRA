"""
model.py
--------
Thin wrapper around RoBERTa-base for sequence classification.
Supports both full fine-tuning and LoRA fine-tuning via inject_lora().
"""

from transformers import RobertaForSequenceClassification, RobertaConfig
from lora import inject_lora
import torch.nn as nn


def build_model(
    num_labels: int = 2,
    model_name: str = "roberta-base",
    mode: str = "lora",          # "lora" | "full"
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list = None,
    lora_init: str = "paper",
    lora_merge_weights: bool = True,
    lora_train_bias: str = "none",
):
    """
    Load pretrained RoBERTa-base and optionally inject LoRA adapters.

    Args:
        num_labels:     number of output classes.
        model_name:     HuggingFace model hub identifier.
        mode:           "lora" injects adapters and freezes base weights;
                        "full" trains all parameters.
        rank:           LoRA rank (ignored in full mode).
        alpha:          LoRA alpha scaling (ignored in full mode).
        dropout:        LoRA internal dropout (ignored in full mode).
        target_modules: which linear layer names to replace with LoraLinear.
        lora_init:      LoRA initialization style ("microsoft" | "paper").
        lora_merge_weights: merge LoRA weights into base on eval().
        lora_train_bias: train bias policy for LoRA mode.

    Returns:
        nn.Module ready for training.
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if mode == "lora":
        model = inject_lora(
            model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            init_method=lora_init,
            merge_weights=lora_merge_weights,
            train_bias=lora_train_bias,
        )
    elif mode == "full":
        # All parameters trainable
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        print(f"[build_model] Full fine-tune: {total:,} trainable params")
    else:
        raise ValueError(f"mode must be 'lora' or 'full', got '{mode}'")

    return model
