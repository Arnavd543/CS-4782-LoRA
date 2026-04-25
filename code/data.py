"""
data.py
-------
Data loading utilities for GLUE benchmarks (SST-2, MRPC).
Uses HuggingFace `datasets` + `transformers` tokenizer.
"""

from datasets import load_dataset
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "sst2": 2,
    "mrpc": 2,
}


def get_tokenizer(model_name: str = "roberta-base") -> RobertaTokenizerFast:
    return RobertaTokenizerFast.from_pretrained(model_name)


def get_dataloaders(
    task: str = "sst2",
    model_name: str = "roberta-base",
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """
    Download (or load cached) GLUE task and return train/validation DataLoaders.

    Args:
        task:        GLUE task name, e.g. "sst2" or "mrpc".
        model_name:  tokenizer to use.
        max_length:  max token sequence length (truncate/pad to this).
        batch_size:  DataLoader batch size.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader, num_labels)
    """
    assert task in TASK_TO_KEYS, f"Unsupported task '{task}'. Choose from {list(TASK_TO_KEYS)}"

    tokenizer = get_tokenizer(model_name)
    key1, key2 = TASK_TO_KEYS[task]

    raw = load_dataset("glue", task)

    def tokenize(batch):
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])
        return tokenizer(
            *texts,
            truncation=True,
            max_length=max_length,
        )

    cols_to_remove = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(tokenize, batched=True, remove_columns=cols_to_remove)

    # SST-2 validation split is called 'validation'; keep label column
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    num_labels = TASK_TO_NUM_LABELS[task]
    print(f"[data] Task={task} | Train={len(tokenized['train']):,} "
          f"| Val={len(tokenized['validation']):,} | Labels={num_labels}")

    return train_loader, val_loader, num_labels
