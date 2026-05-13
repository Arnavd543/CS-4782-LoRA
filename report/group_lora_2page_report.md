**LoRA: Low-Rank Adaptation of Large Language Models**

Authors: Rahul B, Amit S, Arnav D

**1. Introduction**

Before LoRA, the task of adapting increasingly large ML models to downstream task via full fine tuning was incredibly expensive as every individual parameter required gradients, optimizer states, and storage. This all changed with the paper "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021) which hypothesized that the changes made during fine tuning had a low intrinsic rank, and could therefore be approximated as a product B*A. Then, the original weight matrix would be frozen and only these two smaller matrices are trained and then added back to the original weights (scaled by alpha/rank). The authors then showed that this matched full fine-tuning across the GLUE benchmark while only having to train less than 1% of parameters. Additionally, additional latency was removed by adding B*A to the original weights after training.

For our project, we reimplemented LoRA from scratch on top of HuggingFace Transformers and evaluated it on SST-2 and MRPC with RoBERTa-base. We then explored four extensions of LoRA — a rank sweep, a target module sweep across attention and FFN matrices, LoRA+ with asymmetric learning rates for A and B, and dropout on the LoRA path — along with a forgetting analysis comparing masked-LM perplexity on wikitext-2 before and after fine tuning.

**2. Chosen Result**

For our reimplementation, we decided to target the RoBERTa-base SST-2 row of Table 2 in the paper, which reported 94.8% accuracy for full fine-tuning and 95.1% accuracy for LoRA when utilizing 8 ranks for the Query and Value projections (using only 0.3M parameters out of 125M). We ultimately chose this metric as it was the paper's headline comparison with full fine tuning and is a core result in their justification.

**3. Methodology**

Implementation: Our LoraLinear class is a subclass of nn.Linear. At construction the original layer's weight is copied and frozen and then two trainable parameters A (rank by input_dim) and B (output_dim by rank) are added after scaling by alpha/rank. We initialized A from a gaussian distribution with mean 0 and standard deviation 0.02 and initialized B to zero, so the weight update starts at zero and training begins from the pretrained behavior. Adapters are injected by walking model.named_modules and replacing matching nn.Linear modules in place. Following loralib, we merge B*A into the base weight on model.eval and unmerge on model.train, giving zero latency overhead at inference time. Only LoRA parameters, the classification head, and the pooler receive gradients while all original parameters are untouched.

Setup: RoBERTa-base on SST-2 (67,349 train / 872 dev) and MRPC (3,668 train / 408 dev), max length 128, batch size 32, AdamW with linear warmup at 6 percent of steps and linear decay, and trained on Google Colab A100 GPU for 10 epochs per run. LoRA uses learning rate 5e-4; full fine-tuning uses 2e-5 (same rates used in original paper). Default LoRA targets are query and value projections at rank 8 with alpha = 8.

Experiments: (1) Baseline reimplementation at rank=8 on both SST-2 and MRPC, comparing full fine tuning to LoRA. (2) Rank sweep on SST-2 across r = 1, 2, 4, 8, 16, 32. (3) Target module sweep at rank=4 across query, key, value, QKV, attention output, FFN-up, and FFN-down. (4) LoRA+ at rank=8 with the learning rate for B set 16 times larger than that of A. (5) LoRA with dropout (p=0.05) on the LoRA path. (6) Forgetting analysis comparing masked-LM perplexity on wikitext-2 for the pretrained model, the full fine-tuned model, and LoRA at ranks 1, 8, and 32.

**4. Results and Analysis**

The results for our experiments are as follows:

| Configuration                  | Trainable params | SST-2 dev acc |
|--------------------------------|-----------------:|--------------:|
| Full fine-tuning               |          124.6 M |    **0.9472** |
| LoRA rank 1 (Q+V)              |           0.63 M |        0.9392 |
| LoRA rank 4 (Q+V)              |           0.74 M |        0.9392 |
| LoRA rank 8 (Q+V)              |  0.89 M (0.71 %) |    **0.9415** |
| LoRA rank 16 (Q+V)             |           1.18 M |        0.9427 |
| LoRA rank 32 (Q+V)             |           1.77 M |        0.9415 |
| LoRA rank 4, FFN_up only       |           0.78 M |    **0.9541** |
| LoRA rank 8 + dropout 0.05     |           0.89 M |        0.9472 |
| LoRA+ rank 8                   |           0.89 M |    **0.9484** |
| MRPC: Full fine-tuning         |          124.6 M |        0.8848 |
| MRPC: LoRA rank 8 (Q+V)        |           0.89 M |        0.8701 |

> **[Figure 1]** Rank sweep — validation accuracy vs LoRA rank on SST-2, with full fine tuning as a horizontal reference. *Insert: `results/rank_sweep/figures/rank_sweep_accuracy.png`*

Parity, achieved. Full fine tuning hit 94.72 percent on SST-2 dev and LoRA rank 8 hit 94.15 percent — a 0.57 point gap while training only 0.71 percent of parameters, within sampling noise of the paper's 94.8 and 95.1. Peak GPU memory dropped from 2.5 GB to 1.3 GB (a 49 percent reduction) under LoRA. MRPC repeats the picture (88.5 vs 87.0). Across the rank sweep, accuracy stays essentially flat from r=1 (93.92) to r=32 (94.15), with r=16 the peak at 94.27, supporting Hu et al.'s finding that very low intrinsic rank suffices on small classification tasks; gains from increasing rank are within seed noise.

> **[Figure 2]** Module comparison — validation accuracy by LoRA target module at rank 4. *Insert: `results/module_comparison/figures/module_comparison_accuracy.png`*

The module sweep gave us our most surprising finding. At rank=4, injecting LoRA into the FFN up-projection (intermediate.dense) alone hit 95.41 percent, beating both Q+V (the paper's default choice) and full fine tuning. Within attention, V outperformed Q which outperformed K (the worst single target at 93.35), matching the ordering reported in the paper.

> **[Figure 3]** Extensions comparison — baseline LoRA vs LoRA+ vs LoRA + Dropout vs LoRA+ + Dropout. *Insert: `results/extensions/figures/extensions_comparison.png`*

LoRA+ was the strongest of our extensions. With lr_B set to 16 times lr_A, LoRA+ reached 94.84 percent, beating both baseline LoRA and full fine tuning. Adding dropout 0.05 to baseline LoRA also helped marginally (94.72 vs 94.15), but combining LoRA+ with dropout regressed to 94.04 percent, likely from over-regularization on the small dev set.

> **[Figure 4]** Masked-LM perplexity drift on wikitext-2 after SST-2 fine tuning. *Insert: `results/forgetting/figures/perplexity_comparison.png`*

LoRA forgets dramatically less than full fine tuning. The pretrained model has wikitext-2 masked-LM perplexity 7.37. After SST-2 fine tuning, full FT catastrophically forgets (5,473, roughly 740 times pretrained), while LoRA stays close to baseline: r=1 at 11.32 (+54%), r=8 at 9.90 (+34%), r=32 at 9.69 (+31%). Interestingly, with alpha fixed at 8, higher rank forgets *less* — the opposite of the naive "more capacity, more drift" intuition. This follows from the alpha/rank scaling: r=1 applies B*A at scaling 8 while r=32 applies it at 0.25, so the effective per-step weight update is largest at the lowest rank.

**5. Reflections**

What we learned. The merge and unmerge mechanism is the unsung hero of LoRA: it is what makes the method drop-in for production rather than a research curiosity. Implementing it correctly required handling the train and eval transition idempotently. We also learned that "trainable parameters" claims are slippery — the paper's 0.3M figure for rank 8 excludes the classification head, which on a new task must be trained from scratch and adds another 0.6M parameters. Including the head is the honest accounting.

What we would do differently. We would run multiple seeds to get meaningful confidence intervals across the rank settings, since the gaps between adjacent ranks (less than 0.4 points) sit inside likely seed variance. We would also sweep the LoRA+ ratio at a fixed effective learning rate to disentangle the asymmetry benefit from the learning rate change, implement AdaLoRA (Zhang et al., 2023) for adaptive rank allocation, and extend the forgetting analysis with a focused alpha sweep at each rank to validate the scaling explanation.

Broader takeaway. Parameter-efficient fine-tuning inverts a familiar deep-learning bargain: a small amount of expressivity is traded for a large amount of practicality. Single-GPU training, tiny checkpoints, fast task-switching, and zero inference latency. For most production fine-tuning, that is the right trade.

**References**

[1] Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[2] Hayou, S., Ghosh, N., Yu, B. (2024). LoRA+: Efficient Low Rank Adaptation of Large Models. arXiv:2402.12354.

[3] Zhang, Q. et al. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. arXiv:2303.10512.

[4] Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.

[5] Wang, A. et al. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. arXiv:1804.07461.
