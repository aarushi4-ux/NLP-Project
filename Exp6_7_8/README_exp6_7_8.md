# Experiments 6, 7 & 8 — Decomposition, Error Analysis, and SOTA Comparison

All three experiments load directly from `exp1_2_results.pt` and require no additional model inference. They can be run in any order after Exp 1/2 completes.

---

## Experiment 6 — FFN vs Attention Layer Decomposition

### What It Measures

Exp 6 determines whether hallucination signals in the cosine-drift layer profile localise preferentially to **FFN sublayers** or **attention sublayers** within GPT-2 Medium, and whether the effect is stronger in early, mid, or late layers.

For each layer, cosine-similarity scores between the component output and the residual stream are extracted (saved during Exp 1/2 via forward hooks) and evaluated with AUROC against the test labels. A point-biserial correlation is also computed per layer for the overall cosine-drift profile.

### Inputs

| Input | Key in `.pt` |
|---|---|
| Per-layer attention cosine scores | `attn_layer_scores` |
| Per-layer FFN cosine scores | `ffn_layer_scores` |
| Pre-computed AUROC profiles | `attn_layer_profile`, `ffn_layer_profile` |
| Per-layer raw cosine drift | `layer_scores_raw` |
| Test labels | `test_labels` |

### How to Run

```bash
python run_exp6.py
```

`exp1_2_results.pt` must be in the working directory.

### Outputs (terminal only)

- Table: FFN vs Attention AUROC per layer band (early / mid / late / mid-to-late)
- Table: Layer-wise FFN and Attention AUROC with winner per layer
- Table: Point-biserial correlation of cosine drift vs label per layer
- Interpretation summary: which component and band dominates

---

## Experiment 7 — Error Analysis (False Positives & False Negatives)

### What It Measures

Exp 7 builds a **composite hallucination score** by averaging all per-token metrics, applies a data-driven threshold (`τ = mean(composite)`), and then categorises every test sample as TP, TN, FP, or FN. It produces:

- Ranked lists of the worst false positives (faithful samples wrongly flagged) and false negatives (hallucinations missed)
- Token-level attribution: which token drove the misclassification and which metric dominated at that position
- Mechanistic natural-language explanations per failure case

### Inputs

| Input | Source |
|---|---|
| Per-token metric scores | `per_token_scores` in `exp1_2_results.pt` |
| Test labels and IDs | `test_labels`, `test_sample_ids` in `.pt` |
| Raw dataset | `../data/qa_data.json` (JSON array or JSONL) |

The raw dataset is used to recover `question` and `response_text` for display. If a sample ID is not found in the dataset, those fields are left blank; scoring still proceeds.

### How to Run

```bash
python run_exp7.py
```

Edit `data_path = "../data/qa_data.json"` at the top of the script if your dataset lives elsewhere. `exp1_2_results.pt` must be in the working directory.

### Outputs

| File | Contents |
|---|---|
| `exp7_cases.csv` | All test samples with score, prediction, and labels |
| `exp7_false_positives.csv` | Faithful samples predicted as hallucinated, sorted by descending score |
| `exp7_false_negatives.csv` | Hallucinated samples predicted as faithful, sorted by ascending score |
| `exp7_report_cases.csv` | 3 enriched failure cases with token attribution and mechanistic explanation |

Terminal output includes the top-3 FPs and FNs and a full confusion matrix (TP / TN / FP / FN, accuracy, precision, recall).

### Threshold

`τ` is set to `mean(composite)` across all test samples, so approximately 50% of samples are predicted hallucinated. This is intentional — the goal is error characterisation, not optimising F1.

---

## Experiment 8 — SOTA Gap Analysis

### What It Measures

Exp 8 contextualises the best AUROC achieved across Exp 1/2 against two published systems:

- **ReDeEP** — AUROC ≈ 0.82
- **LUMINA** — AUROC ≈ 0.87

The *gap closed* metric measures how much of the distance from the Attention Entropy baseline (B1) to SOTA has been recovered:

```
gap_closed = (your_AUROC − baseline_AUROC) / (sota_AUROC − baseline_AUROC)
```

### Inputs

| Input | Key in `.pt` |
|---|---|
| Individual metric AUROCs | `individual` |
| Composite step AUROCs | `composite` |
| Baseline (Attention Entropy B1) | `individual["Attention Entropy (B1)"]["AUROC"]` |

### How to Run

```bash
python run_exp8.py
```

`exp1_2_results.pt` must be in the working directory.

### Outputs (terminal only)

```
===== E8 RESULTS =====

Baseline (Attention Entropy): 0.XXXX
Best Method: composite -> CIE top-3 layers
Best AUROC: 0.XXXX

SOTA Comparison:
ReDeEP (~0.82): Gap Closed = 0.XXXX (XX.XX%)
LUMINA (~0.87): Gap Closed = 0.XXXX (XX.XX%)
```

The best method is selected automatically by scanning both `individual` and `composite` sections of the checkpoint for the highest AUROC.

---

## Shared Dependency

All three experiments depend solely on `exp1_2_results.pt`. No GPU or model inference is required. Typical runtime is under 30 seconds each on CPU.
