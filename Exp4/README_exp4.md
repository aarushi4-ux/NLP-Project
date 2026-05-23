# Experiment 4 — Temporal Precedence of Representation Drift

## What This Experiment Measures

Experiment 4 tests whether representation anomalies **precede** hallucination onset in the token stream, or only appear coincidentally at the hallucinated token itself.

For each hallucinated sample, the experiment identifies the onset token `t` (first hallucinated position) and measures five metrics at positions `t−3, t−2, t−1, t, t+1`. If drift signals consistently peak *before* `t`, this is evidence that the model's internal representations begin to diverge from the faithful manifold several tokens ahead of the surface error — suggesting the metrics could be used for early warning.

Statistical significance is tested with Mann-Whitney U comparing pooled pre-onset (`t−3 + t−2`) against post-onset (`t+1`) distributions.

---

## Files

| File | Role |
|---|---|
| `run_e4_new.py` | Main script — loads per-token scores from `exp1_2_results.pt`, aligns them to onset positions from `e4_processed.json`, computes windowed means, runs statistics, and saves a plot |

`temporal.py` is an earlier prototype that operates directly on the raw activation tensors saved by Exp 3. `run_e4_new.py` is the current canonical version and runs without needing the `activations/` directory.

---

## Inputs

| Input | Description |
|---|---|
| `../outputs/exp1_2_results.pt` | Frozen Exp 1/2 checkpoint — provides `per_token_scores` and `test_sample_ids` |
| `e4_processed.json` | JSON list of processed samples; each entry has a `"t"` field (list of onset token indices) and a `"labels"` field (per-token binary label) |

### `e4_processed.json` expected format

```json
[
  {
    "t": [42, null],
    "labels": [0, 0, 0, 1, 1, 0],
    "seq_len": 6
  },
  ...
]
```

`"t"` entries may be `null` for samples with no detected onset; those are skipped.

---

## Outputs

| Output | Description |
|---|---|
| `../outputs/e4_temporal_plot.png` | Line plot of min-max normalised metric values across the `[t−3, t−2, t−1, t, t+1]` window, with shaded ±1 SD bands and onset marker |
| Terminal table | Mean metric values per offset position |
| Terminal stats | Mann-Whitney U test results (pre vs post onset) |

---

## How to Run

```bash
python run_e4_new.py
```

Paths are hardcoded at the top of the script:

```python
data = torch.load("../outputs/exp1_2_results.pt", ...)
with open("e4_processed.json") as f: ...
```

Edit these if your directory layout differs.

---

## Metrics Analysed

| Key in script | Metric name in `.pt` |
|---|---|
| `cosine` | `Cosine Drift` |
| `mahalanobis` | `Mahalanobis` |
| `logit_lens` | `Logit Lens Div.` |
| `pca_dev` | `PCA Deviation` |
| `cie` | `CIE top-3 layers` |

---

## Interpretation Guide

- **Peak at t−2 or t−1** → metric detects hallucination before it surfaces; useful for early intervention.
- **Peak at t** → metric is reactive, not predictive; still useful for detection but not anticipation.
- **Mann-Whitney p < 0.05** → statistically significant difference between pre- and post-onset distributions.

The plot normalises each metric to `[0, 1]` (min-max) so that all five can be compared on the same axis despite different scales. The vertical red dashed line marks `t = 0` (onset).
