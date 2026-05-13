from transformers import RobertaForSequenceClassification
from lora import inject_lora


def build_model(
    num_labels: int = 2,
    model_name: str = "roberta-base",
    mode: str = "lora",
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list = None,
    lora_init: str = "paper",
    lora_merge_weights: bool = True,
    lora_train_bias: str = "none",
):
    # Load pretrained RoBERTa-base and optionally inject LoRA adapters.
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
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        print("[build_model] Full fine-tune: " + format(total, ",") + " trainable params")
    else:
        raise ValueError(f"mode must be 'lora' or 'full', got '{mode}'")

    return model
