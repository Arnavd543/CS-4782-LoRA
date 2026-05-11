from transformers import RobertaForSequenceClassification
from lora import inject_lora


def build_model(
    numLabels: int = 2,
    modelName: str = "roberta-base",
    mode: str = "lora",
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    targetModules: list = None,
    loraInit: str = "paper",
    loraMergeWeights: bool = True,
    loraTrainBias: str = "none",
):
    # Load pretrained RoBERTa-base and optionally inject LoRA adapters.
    if targetModules is None:
        targetModules = ["query", "value"]

    model = RobertaForSequenceClassification.from_pretrained(
        modelName,
        num_labels=numLabels,
    )

    if mode == "lora":
        model = inject_lora(
            model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            targetModules=targetModules,
            initMethod=loraInit,
            mergeWeights=loraMergeWeights,
            trainBias=loraTrainBias,
        )
    elif mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        print("[build_model] Full fine-tune: " + format(total, ",") + " trainable params")
    else:
        raise ValueError(f"mode must be 'lora' or 'full', got '{mode}'")

    return model
