# Experiment 5 — Cross-Domain Generalisation (HaluEval)

## What This Experiment Measures

Experiment 5 tests whether the metrics fitted on RAGTruth generalise to a different hallucination dataset without any retraining — a **zero-shot cross-domain transfer** evaluation.

The fitted Mahalanobis and PCA parameters from `exp1_2_results.pt` are applied directly to hidden states extracted from HaluEval (`general_data.json`). Individual and composite AUROCs are compared against the RAGTruth baseline to quantify the generalisation drop.

---

## Files

| File | Role |
|---|---|
| `haluevaltest.py` | Self-contained script — loads HaluEval, extracts hidden states with GPT-2 Medium, scores with frozen Exp 1/2 parameters, prints AUROC table and cross-domain summary |

---

## Inputs

| Input | Description |
|---|---|
| HaluEval `general_data.json` | JSONL file; each line has `user_query`, `chatgpt_response`, and `hallucination` (yes/no or 0/1) |
| `exp1_2_results.pt` | Frozen checkpoint from Exp 1/2 — provides `mahalanobis.mu`, `mahalanobis.inv_cov`, `pca.components`, `pca.mean` |

**Critical constraint:** The Mahalanobis and PCA parameters must **not** be re-fitted on HaluEval. They must be loaded directly from the checkpoint so the evaluation is zero-shot.

---

## Outputs

All output is printed to the terminal:

```
=== HALUEVAL RESULTS ===

cosine     | AUROC: 0.XXXX
mahal      | AUROC: 0.XXXX
logit      | AUROC: 0.XXXX
pca        | AUROC: 0.XXXX
cie        | AUROC: 0.XXXX

COMPOSITE AUROC: 0.XXXX

=== EXP 5 CROSS-DOMAIN SUMMARY ===
RAGTruth baseline : 0.XXXX
HaluEval composite: 0.XXXX
Drop              : 0.XXXX
```

---

## How to Run

1. Update the data path at the top of `haluevaltest.py`:

```python
path = "/path/to/general_data.json"
```

2. Ensure `exp1_2_results.pt` is in the working directory (or update the `torch.load` path).

3. Run:

```bash
python haluevaltest.py
```

**Device:** The extractor defaults to MPS (Apple Silicon) if available, then CUDA, then CPU. Edit `HiddenStateExtractor(device="...")` to override.

---

## Metrics

The five scoring functions applied to hidden states of shape `(n_layers, seq_len, hidden_dim)`:

| Name | Method |
|---|---|
| `cosine` | Mean L2 norm of hidden states across layers (proxy for activation magnitude drift) |
| `mahal` | Mahalanobis distance from the faithful distribution μ, using frozen `inv_cov` per layer |
| `logit` | Mean variance of hidden states across layers (proxy for representation uncertainty) |
| `pca` | Squared projection onto the faithful PCA subspace (reconstruction energy) |
| `cie` | Mean absolute value of hidden states (proxy for activation magnitude) |

**Composite** is a fixed weighted sum:

```python
0.25 * cosine + 0.25 * mahal + 0.20 * logit + 0.15 * pca + 0.15 * cie
```

---

## Notes

- The `ragtruth_auc` field is read from the checkpoint if present. If missing, the cross-domain drop cannot be computed and a warning is printed.
- `safe_auc` returns `nan` if the score array is constant or labels are single-class, so degenerate runs are handled gracefully.
- Hidden states are trimmed to 24 layers (removing the embedding layer at index 0 if `n_layers > 24`).
