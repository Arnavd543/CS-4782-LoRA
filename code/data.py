# GLUE task metadata shared by train.py and evaluate.py.
# get_dataloaders below is only used by the standalone evaluate.py utility;
# train.py loads data through HuggingFace Trainer with its own DataCollator.

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
    # Return (train_loader, val_loader, num_labels) for a GLUE task. Used by evaluate.py only.
    assert task in TASK_TO_KEYS, f"Unsupported task '{task}'. Choose from {list(TASK_TO_KEYS)}"

    tokenizer = get_tokenizer(model_name)
    key1, key2 = TASK_TO_KEYS[task]

    raw = load_dataset("glue", task)

    def tokenize(batch):
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])
        return tokenizer(*texts, truncation=True, max_length=max_length)

    cols_to_remove = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(tokenize, batched=True, remove_columns=cols_to_remove)

    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    num_labels = TASK_TO_NUM_LABELS[task]
    print("[data] Task=" + str(task) + " | Train=" + format(len(tokenized["train"]), ",") +
          " | Val=" + format(len(tokenized["validation"]), ",") + " | Labels=" + str(num_labels))

    return train_loader, val_loader, num_labels
