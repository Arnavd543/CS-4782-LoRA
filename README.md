# LoRA: Low-Rank Adaptation of Large Language Models — Replication & Extensions

A from-scratch re-implementation of [Hu et al. 2021](https://arxiv.org/abs/2106.09685) on top of
HuggingFace Transformers, with four extensions, evaluated on GLUE / SST-2 and MRPC using
RoBERTa-base. Final project for CS 4782 (Spring 2026).

**Authors:** Rahul B, Amit S, Arnav D

---

## 1. Introduction

This repository contains a re-implementation of *LoRA: Low-Rank Adaptation of Large Language Models*
(Hu et al., 2021), the paper that introduced the now-dominant parameter-efficient fine-tuning method
for large pretrained transformers. The key idea is to freeze the pretrained weight `W₀` and learn a
low-rank update `ΔW = BA` (with `A ∈ ℝʳˣᵈ`, `B ∈ ℝᵈˣʳ`, `r ≪ d`), scaled by `α / r`. Training
under 1% of parameters reaches full fine-tuning accuracy on GLUE, and at inference `BA` merges back
into `W₀` so there is no extra latency.

## 2. Chosen Result

We target the **RoBERTa-base SST-2 row of Table 2** in the paper: full fine-tuning at **94.8%**
and LoRA r=8 (Q+V projections, 0.3M trainable parameters out of 125M) at **95.1%**. This is the
paper's headline comparison and the single number that justifies its central claim — that low-rank
updates suffice to match full fine-tuning. We reproduce this baseline and then run four extensions
on top.

(more interesting results and graphs/figures in here: https://drive.google.com/drive/folders/1QPv1wVYnolIMqpn9FnnjaUo0Etm-xyBo?usp=sharing)

## 3. GitHub Contents

```
CS-4782-LoRA/
├── README.md                       # This file
├── LICENSE                         # MIT
├── .gitignore
├── code/
│   ├── requirements.txt
│   ├── lora.py                     # LoraLinear layer + inject_lora() entry point
│   ├── model.py                    # RoBERTa-base wrapper (LoRA or full FT mode)
│   ├── data.py                     # GLUE task constants + standalone DataLoader helper
│   ├── train.py                    # Training loop, W&B logging, Trainer wrapper
│   ├── analyze.py                  # Aggregates per-run JSON logs into CSVs and figures
│   ├── forgetting.py               # Masked-LM perplexity probe on wikitext-2
│   ├── evaluate.py                 
│   ├── Implement_LORA.ipynb        # End-to-end notebook driving every experiment on Colab
│   └── configs/
│       ├── baseline.yaml           # Full fine-tune (lr 2e-5)
│       ├── lora_r8.yaml            # Paper-style LoRA (r=8, α=8, Q+V, paper init)
│       ├── lora_plus_r8.yaml       # LoRA+ (asymmetric lr, lr_B = 16 · lr_A)
│       ├── lora_dropout_r8.yaml    # LoRA + dropout p=0.05 on the low-rank path
│       └── lora_plus_dropout_r8.yaml
├── data/
│   └── README.md                   # Dataset acquisition (HuggingFace datasets, auto-download)
├── results/
│   ├── baseline/ rank_sweep/ module_comparison/ extensions/ mrpc/ forgetting/
│   │   └── figures/                # Per-experiment plots
│   ├── logs/                       # Per-run JSON metrics
│   ├── tables/                     # CSV summaries (more in drive)
│   └── figures/                    # Cross-experiment overview plots (more in drive)
├── poster/
│   └── LORA Presentation Poster.pdf
└── report/
    └── group_lora_2page_report.md  
```

## 4. Re-implementation Details

**Model & datasets.** RoBERTa-base on SST-2 (67,349 train / 872 dev) and MRPC (3,668 train / 408 dev),
loaded via `datasets.load_dataset("glue", task)`. Max sequence length 128, batch size 32.

**LoraLinear layer** (`code/lora.py`):
- Subclasses `nn.Linear`, copies and freezes the pretrained weight at construction.
- Adds trainable `A ∈ ℝʳˣᵈⁱⁿ` (init `N(0, 0.02)` per paper §4) and `B ∈ ℝᵈᵒᵘᵗˣʳ` (init zero).
- `forward(x)` returns `W₀x + (α / r) · BAx`.
- On `model.eval()`, merges `BA` into `W₀`; on `model.train()`, un-merges. Zero added inference
  latency.
- `inject_lora(model, rank, alpha, target_modules, ...)` walks the model tree and replaces matching
  `nn.Linear` modules in place. Only LoRA parameters, the classification head, and the pooler
  receive gradients.

**Training.** AdamW with linear warmup (6% of steps) and linear decay, 10 epochs per run. LoRA
learning rate 5e-4, full fine-tuning 2e-5 (paper hyperparameters). Mixed-precision (fp16) when a
CUDA device is available. Driven by `train.py` via per-experiment YAML configs in `code/configs/`.

**Extensions.**
1. **Rank sweep** — `r ∈ {1, 2, 4, 8, 16, 32}` at fixed `α = 8`.
2. **Target-module sweep** at `r = 4` — Wq, Wk, Wv, QKV, attention-output, FFN-up, FFN-down.
3. **LoRA+** (Hayou et al., 2024) — asymmetric learning rates, `lr_B = 16 · lr_A`.
4. **LoRA + dropout** — `Dropout(p=0.05)` on the low-rank path.
5. **Forgetting analysis** — masked-LM perplexity on wikitext-2 before vs after SST-2
   fine-tuning, for the pretrained model, full FT, and LoRA at r=1/8/32 (`code/forgetting.py`).

**Evaluation metrics.** Validation accuracy on the GLUE dev split (primary). Trainable parameter
count, peak GPU memory, and wall-clock time are also logged per run.

## 5. Reproduction Steps

### Environment

```bash
git clone https://github.com/Arnavd543/CS-4782-LoRA.git
cd CS-4782-LoRA
pip install -r code/requirements.txt
```

Tested on Google Colab with an A100 GPU. CPU works for tiny smoke tests; a single training run
needs roughly 15–20 minutes per 10 epochs on an A100.

### Reproducing all experiments (Colab)

Open `code/Implement_LORA.ipynb` in Google Colab with an A100 (or T4 with longer runtimes) and run
all cells. The notebook mounts Google Drive and writes checkpoints/results directly to a
`lora_results/` folder via the `--checkpoints_dir` and `--results_dir` flags, so nothing needs to
be copied back after each run.

### Running individual scripts

```bash
# Baseline runs
python code/train.py --config code/configs/baseline.yaml --run_name baseline_full_ft
python code/train.py --config code/configs/lora_r8.yaml  --run_name baseline_lora_r8_paper

# Rank sweep
for r in 1 2 4 8 16 32; do
  python code/train.py --config code/configs/lora_r8.yaml --rank $r --run_name lora_rank_$r
done

# Target-module sweep at r=4
python code/train.py --config code/configs/lora_r8.yaml --rank 4 --target_modules query --run_name lora_module_Wq
# (...and analogous calls for key/value/QKV/intermediate.dense/output.dense/attention.output.dense)

# Extensions
python code/train.py --config code/configs/lora_plus_r8.yaml          --run_name lora_plus_r8
python code/train.py --config code/configs/lora_dropout_r8.yaml       --run_name lora_dropout_r8
python code/train.py --config code/configs/lora_plus_dropout_r8.yaml  --run_name lora_plus_dropout_r8

# Aggregate per-run JSON logs into CSVs and figures
python code/analyze.py

# Forgetting probe (requires the trained checkpoints listed above)
python code/forgetting.py
```

All scripts accept `--checkpoints_dir` and `--results_dir` so you can point them at Google Drive
directly instead of writing to a local `checkpoints/` and `results/` tree.

### Hardware

| Component | Used |
|---|---|
| GPU | NVIDIA A100 (Colab Pro) |
| Time per full FT run | ~19 minutes (10 epochs, SST-2) |
| Time per LoRA r=8 run | ~17 minutes (10 epochs, SST-2) |
| Peak memory: full FT | ~2.5 GB |
| Peak memory: LoRA r=8 | ~1.3 GB |

## 6. Results / Insights

| Configuration | Trainable params | SST-2 dev acc |
|---|---:|---:|
| Full fine-tuning | 124.6 M | **0.9472** |
| LoRA r=8 (Q+V, paper init) | 0.89 M (0.71%) | **0.9415** |
| LoRA r=1 (Q+V) | 0.63 M | 0.9392 |
| LoRA r=16 (Q+V) | 1.18 M | 0.9427 |
| LoRA r=32 (Q+V) | 1.77 M | 0.9415 |
| LoRA r=4, FFN_up only | 0.78 M | **0.9541** |
| LoRA+ r=8 (lr_B / lr_A = 16) | 0.89 M | **0.9484** |
| LoRA r=8 + dropout 0.05 | 0.89 M | 0.9472 |
| MRPC: Full FT | 124.6 M | 0.8848 |
| MRPC: LoRA r=8 | 0.89 M | 0.8701 |

Key findings:

- **Headline replication.** Full FT 94.72% vs LoRA r=8 94.15% — within sampling noise of the
  paper's 94.8 / 95.1 at 0.71% of trainable parameters and ~49% less peak GPU memory. The MRPC
  baselines repeat the picture (88.5 vs 87.0).
- **Rank is essentially flat.** From r=1 to r=32, accuracy varies between 93.92% and 94.27%; r=1
  alone is within 0.23 pp of r=8 using 0.5% of parameters. Strongly supports the "intrinsic rank
  is low" claim.
- **FFN_up wins the module sweep.** At r=4, LoRA on the FFN up-projection alone reaches **95.41%**,
  beating both Q+V and full fine-tuning. Suggests the MLP up-projection is at least as valuable
  a target as Q+V on this task.
- **LoRA+ is the strongest extension.** With `lr_B = 16 · lr_A`, LoRA+ hits **94.84%**, exceeding
  both baseline LoRA and full FT.
- **LoRA forgets less than full FT.** Wikitext-2 masked-LM perplexity goes from 7.37 (pretrained)
  to ~5,400 after full FT, but stays ≤ 12 for every LoRA variant. With `α` fixed, *higher* LoRA
  rank forgets *less* — opposite of the naive intuition, and a direct consequence of the `α / r`
  scaling factor.

See `results/` for full CSVs and figures, and `report/group_lora_2page_report.md` for the written
2-page summary.

## 7. Conclusion

The paper's central claim reproduces cleanly on RoBERTa-base / SST-2 with a from-scratch
implementation: LoRA matches full fine-tuning to within sampling noise while training under 1% of
parameters and cutting peak GPU memory roughly in half. Beyond the replication, two of our
extensions produced standalone findings worth flagging — LoRA+ beats both baseline LoRA and full
fine-tuning (94.84%), and a single-module LoRA on the FFN up-projection at r=4 reaches 95.41%,
beating every attention-only target. The forgetting probe confirms that low-rank ΔW preserves
masked-LM ability that full fine-tuning destroys, with the additional twist that higher rank
forgets less when α is held fixed — pointing to ΔW magnitude rather than rank as the operative
mechanism. Natural next steps are multiple-seed runs, a LoRA+ ratio sweep at fixed effective
learning rate, and an AdaLoRA-style adaptive-rank implementation.

## 8. References

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Hayou, S., Ghosh, N., Yu, B. (2024). *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv:2402.12354.
- Zhang, Q. et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* arXiv:2303.10512.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Wang, A. et al. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* arXiv:1804.07461.
- HuggingFace `transformers`, `datasets`, `accelerate` (https://huggingface.co/docs).
- Microsoft `loralib` reference implementation (https://github.com/microsoft/LoRA) — referenced for
  the merge/un-merge pattern; our code is implemented from scratch.

## 9. Acknowledgements

This project was completed as part of **CS 4782: Deep Learning** at Cornell University (Spring
2026). We thank the course staff for guidance on the replication-with-extensions format and Google
Colab Pro for the A100 compute that made the full-fine-tuning runs possible.
