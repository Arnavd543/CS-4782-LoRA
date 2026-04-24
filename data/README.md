# Data

All datasets are downloaded automatically via the HuggingFace `datasets` library.
**No manual download required.**

## Datasets used

| Dataset | GLUE task key | Task type | Train size | Val size | Source |
|---------|--------------|-----------|-----------|----------|--------|
| SST-2   | `sst2`       | Sentiment classification | 67,349 | 872 | Stanford |
| MRPC    | `mrpc`       | Paraphrase detection | 3,668 | 408 | Microsoft |

## Automatic download

```python
from datasets import load_dataset

sst2 = load_dataset("glue", "sst2")
mrpc = load_dataset("glue", "mrpc")
```

Datasets are cached to `~/.cache/huggingface/datasets/` by default.
In Google Colab, set a custom cache path to persist across sessions:

```python
import os
os.environ["HF_DATASETS_CACHE"] = "/content/drive/MyDrive/hf_cache"
```

## License

- **SST-2**: Stanford Sentiment Treebank — for non-commercial research use.
- **MRPC**: Microsoft Research Paraphrase Corpus — for research use only.
- **GLUE Benchmark**: [gluebenchmark.com](https://gluebenchmark.com/) — see individual task licenses.

## Citation

```bibtex
@inproceedings{wang2018glue,
  title={GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and others},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}
```
