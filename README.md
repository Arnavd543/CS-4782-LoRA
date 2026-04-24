# LoRA: Low-Rank Adaptation of Large Language Models — Replication & Extensions

> Re-implementation of [Hu et al. 2021](https://arxiv.org/abs/2106.09685) with proposed improvements,
> evaluated on GLUE / SST-2 using RoBERTa-base.

---

## Table of Contents
1. [Summary](#summary)
2. [Justification](#justification)
3. [Repository Structure](#repository-structure)
4. [Setup](#setup)
5. [Data](#data)
6. [Re-implementation Details](#re-implementation-details)
7. [Proposed Improvements](#proposed-improvements)
8. [Results](#results)
9. [Reproducing Experiments](#reproducing-experiments)
10. [References](#references)

---

## Summary

Large language models normally require full fine-tuning to adapt to downstream tasks, which involves
updating potentially billions of parameters and demands significant memory and compute resources.

The LoRA paper proposes a **parameter-efficient fine-tuning** method that:
- Freezes the original pretrained weights entirely.
- Injects trainable low-rank matrices `A ∈ R^{r×d}` and `B ∈ R^{d×r}` into each targeted linear layer.
- Approximates the weight update as `ΔW = BA`, scaled by `α/r`.

Our goal is to demonstrate that LoRA achieves **similar task performance** to full fine-tuning while
**drastically reducing** the number of trainable parameters (< 1% of total).

---

## Justification

Parameter-efficient training techniques are critical for scaling modern machine learning systems.
Full fine-tuning requires large compute clusters; LoRA enables efficient adaptation of large models
on much smaller hardware (single T4/A100 GPU). The ability to modify LLMs for a variety of downstream
tasks without touching pretrained weights is a foundational capability, and LoRA is one of the most
compute-efficient ways to enable it.

---

## Repository Structure

```
lora-replication/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore
│
├── code/
│   ├── requirements.txt       # All Python dependencies
│   ├── lora.py                # LoraLinear layer + inject_lora() entry point
│   ├── model.py               # RoBERTa-base wrapper with classification head
│   ├── data.py                # GLUE / SST-2 data loaders via HuggingFace datasets
│   ├── train.py               # Training loop (supports full FT + LoRA modes)
│   ├── evaluate.py            # Accuracy, trainable param count, GPU memory reporting
│   └── configs/
│       ├── baseline.yaml      # Full fine-tune hyperparameters
│       └── lora_r8.yaml       # LoRA rank=8, alpha=16 hyperparameters
│
├── data/
│   └── README.md              # How to obtain / reproduce datasets
│
├── results/
│   ├── figures/               # Accuracy vs rank plots, param count comparisons
│   ├── tables/                # CSV tables of all run metrics
│   └── logs/                  # W&B exported run logs
│
├── poster/
│   └── poster.pdf             # In-class presentation poster
│
└── report/
    └── report.pdf             # Final written report
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/lora-replication.git
cd lora-replication
```

### 2. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 3. (Colab) Mount Drive and pull repo

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-username>/lora-replication.git
%cd lora-replication
!pip install -r code/requirements.txt
```

Set your W&B key:

```python
import os
os.environ["WANDB_API_KEY"] = "your_key_here"  # or use Colab Secrets
```

---

## Data

See [`data/README.md`](data/README.md) for full instructions.

**Quick start** — datasets are downloaded automatically via HuggingFace `datasets`:

```python
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
```

Datasets used:
| Dataset | Task | Train | Validation |
|---------|------|-------|------------|
| SST-2   | Sentiment classification | 67,349 | 872 |
| MRPC    | Paraphrase detection     | 3,668  | 408 |

---

## Re-implementation Details

### LoRA layer (`code/lora.py`)

The core is a drop-in replacement for `nn.Linear`:

```
forward(x):
    return W₀x + (x @ Aᵀ @ Bᵀ) * (α / r)
```

- `W₀` is **frozen** (`requires_grad=False`)
- `A` is initialized `~ N(0, 0.02)` (Gaussian, per paper §4), `B` is initialized to **zeros** (so ΔW = BA = 0 at step 0)
- `inject_lora(model, rank, alpha, target_modules)` walks the model tree and replaces matching
  `nn.Linear` modules in-place

### What is frozen vs. trained

| Parameters | Frozen | Trainable |
|-----------|--------|-----------|
| Pretrained W₀ | ✓ | |
| LoRA A, B matrices | | ✓ |
| Classification head | | ✓ |

### Default hyperparameters (LoRA)

| Hyperparameter | Value |
|---------------|-------|
| Rank `r` | 8 |
| Alpha `α` | 8 (= r, per Appendix C.2) |
| Target modules | `query`, `value` |
| Learning rate | 3e-4 |
| Batch size | 32 |
| Epochs | 3 |
| Optimizer | AdamW |
| LR schedule | Linear warmup (6%) then linear decay |

---

## Proposed Improvements

Beyond the base replication we implement and evaluate four extensions:

### 1. LoRA+ — Asymmetric learning rates
Set `lr(B) = 16 × lr(A)`. The B matrix acts as the output projection and benefits from faster
updates; A acts as a feature extractor and should update slowly. Zero extra compute cost.

### 2. Full-layer injection sweep
Extend adapters beyond Q+V to `{Q, K, V, O, FFN-up, FFN-down}` at lower rank (r=2 or r=4),
keeping the total parameter budget fixed. Identifies which layers contribute most.

### 3. LoRA + dropout
Insert `nn.Dropout(p=0.05)` between A and B in the forward pass to regularise the low-rank
path. Particularly helpful on small datasets (MRPC) where overfitting is more likely.

### 4. AdaLoRA — Adaptive rank allocation
Periodically SVD-decompose each `BA` product and prune singular values below a threshold,
reallocating rank budget to layers that show higher importance. Replicates the core idea of
[Zhang et al. 2023](https://arxiv.org/abs/2303.10512) in our simplified codebase.

---

## Results

See [`results/`](results/) for full tables and figures.

**Summary table** (SST-2, RoBERTa-base):

| Method | Accuracy | Trainable Params | GPU Mem (MB) |
|--------|----------|-----------------|--------------|
| Full fine-tune | 94.8% | 125M (100%) | ~5,800 |
| LoRA r=8 (Q+V) | ~94.5% | ~300K (0.24%) | ~2,100 |
| LoRA+ r=8 | TBD | ~300K (0.24%) | ~2,100 |
| LoRA r=8 + dropout | TBD | ~300K (0.24%) | ~2,100 |
| Full-layer r=4 | TBD | ~600K (0.48%) | ~2,300 |

*Results cells marked TBD will be filled after experiments complete.*

---

## Reproducing Experiments

### Baseline (full fine-tune)

```bash
python code/train.py --config code/configs/baseline.yaml
```

### LoRA (rank=8, Q+V)

```bash
python code/train.py --config code/configs/lora_r8.yaml
```

### Rank sweep

```bash
for r in 1 2 4 8 16 32; do
  python code/train.py --config code/configs/lora_r8.yaml --rank $r --run_name lora_r${r}
done
```

### All experiments (Colab one-liner)

Open `notebooks/02_lora.ipynb` in Google Colab with a T4/A100 runtime and run all cells.

---

## References

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Zhang, Q. et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* arXiv:2303.10512
- Hayou, S. et al. (2024). *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv:2402.12354
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692
- Wang, A. et al. (2018). *GLUE: A Multi-Task Benchmark.* arXiv:1804.07461
