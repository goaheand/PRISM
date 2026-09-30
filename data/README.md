---
license: mit
task_categories:
- other
tags:
- anomaly-detection
- root-cause-localization
- multi-dimensional-data
- AIOps
size_categories:
- 1M<n<10M
---

# NFC: A Multi-Dimensional Root Cause Localization Dataset

**NFC** is a benchmark dataset for **root cause localization on multi-dimensional data**, built from anonymized NFC (Near-Field Communication) transaction records. Each dimension combination carries categorical attributes, and anomalies manifest as unexpected concentrations of records under specific attribute combinations. The task: given a set of anomalous records, identify the attribute combination (e.g., `B=273 & D=42`) that best explains them.

This dataset was used to evaluate **Prism** (**PR**ecision-driven **I**terative **S**earch for **M**ulti-dimensional root cause localization), a root cause localization algorithm that formulates the problem as a precision–recall joint optimization over attribute-selection weights.

## Dataset Structure

| File | Size | Description |
|------|------|-------------|
| `df.csv` | ~146 MiB | Main data table: 5,135,932 records × 15 categorical dimensions |
| `label.json` | ~6 MB | 100 labeled test cases with ground-truth root causes |

### `df.csv`

- 15 anonymized categorical dimensions, named `A` through `O`.
- Values are integer-encoded; missing values are left empty (a dedicated missing-value category).
- 0-based row positions serve as record indices (the default `RangeIndex` after `pd.read_csv`).

Example:

```
A,B,C,D,E,F,G,H,I,J,K,L,M,N,O
3,3,,,3,3,3,,3,3,3,,,3,3
3,3,,,3,3,3,357,3,3,3,,,3,3
```

### `label.json`

A JSON array of 100 test cases. Each case contains:

- **`label`** — the ground-truth root cause, a list of attribute combinations of the form `{"column": "value"}` (values are the integer codes from `df.csv`):

  ```json
  [{"B": "273"}]
  ```

- **`anomalyIndex`** — row indices into `df.csv` of the records flagged as anomalous in this case (typically a few hundred indices).

## Task Definition

For each test case, given `anomalyIndex`, a method must predict one or more root causes as attribute combinations. Predictions are compared against the ground-truth `label`:

- **Precision** — fraction of predicted combinations that are true root causes
- **Recall** — fraction of true root causes that are predicted
- **F1** — harmonic mean

Aggregating TP/FP/FN across all 100 cases yields micro-averaged Precision/Recall/F1.

## Quick Start

```python
import json
import pandas as pd
from huggingface_hub import hf_hub_download

df = pd.read_csv(
    hf_hub_download("Ysqq/NFC", "df.csv", repo_type="dataset")
)

with open(
    hf_hub_download("Ysqq/NFC", "label.json", repo_type="dataset")
) as f:
    cases = json.load(f)

case = cases[0]
anomaly_rows = df.iloc[case["anomalyIndex"]]
print(f"Ground-truth root cause: {case['label']}")
```

A reference implementation (the Prism algorithm with a statistical noise-filtering variant, `PrismNFC`) is available in the accompanying code repository.

## Citation

If you use this dataset in your research, please cite the associated Prism paper (citation to be added upon publication).