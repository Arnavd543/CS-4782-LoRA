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


def get_tokenizer(modelName: str = "roberta-base") -> RobertaTokenizerFast:
    return RobertaTokenizerFast.from_pretrained(modelName)


def get_dataloaders(
    task: str = "sst2",
    modelName: str = "roberta-base",
    maxLength: int = 128,
    batchSize: int = 32,
    numWorkers: int = 2,
):
    # Return (trainLoader, valLoader, numLabels) for a GLUE task.
    assert task in TASK_TO_KEYS, f"Unsupported task '{task}'. Choose from {list(TASK_TO_KEYS)}"

    tokenizer = get_tokenizer(modelName)
    key1, key2 = TASK_TO_KEYS[task]

    raw = load_dataset("glue", task)

    def tokenize(batch):
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])
        return tokenizer(*texts, truncation=True, max_length=maxLength)

    colsToRemove = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(tokenize, batched=True, remove_columns=colsToRemove)

    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainLoader = DataLoader(
        tokenized["train"],
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    valLoader = DataLoader(
        tokenized["validation"],
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    numLabels = TASK_TO_NUM_LABELS[task]
    print("[data] Task=" + str(task) + " | Train=" + format(len(tokenized["train"]), ",") +
          " | Val=" + format(len(tokenized["validation"]), ",") + " | Labels=" + str(numLabels))

    return trainLoader, valLoader, numLabels
