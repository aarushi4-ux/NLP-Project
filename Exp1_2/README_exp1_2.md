# Track B — Experiments 1 & 2
## Pre-Generation Causal Drift Metric | GPT-2-medium | RAGTruth

---

## Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Parses `response.jsonl` + `source_info.jsonl`, splits into train/val/test |
| `extractor.py` | Loads GPT-2-medium, extracts hidden states + attentions + logits per token |
| `metrics.py` | All 5 Track B metrics (+ 2 baselines) |
| `composite.py` | Experiments 1 & 2 end-to-end: individual → incremental composite → AUROC table |
| `plot_results.py` | Layer localisation profile + composite bar chart (required for 4/4) |

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run

```bash
# Experiments 1 & 2
python composite.py --response /path/to/response.jsonl --source /path/to/source_info.jsonl

# After composite.py finishes, generate plots
python plot_results.py
```

---

## What the code does (for evaluator questions)

### Data loading (`data_loader.py`)
- Reads both `.jsonl` files line-by-line
- Joins on `source_id` to attach retrieved context to each response
- Parses annotation field to get binary label (1 = hallucinated) and hallucination type

### Hidden state extraction (`extractor.py`)
- Loads `gpt2-medium` with `output_hidden_states=True, output_attentions=True`
- Forward-passes the full prompt (context + question + answer) in one shot
- Returns a `HiddenStateBundle` with:
  - `hidden_states`: shape `(25, seq_len, 1024)` — layer 0 is embedding, layers 1-24 are transformer blocks
  - `attentions`: shape `(24, 16, seq_len, seq_len)`
  - `logits`: shape `(seq_len, 50257)`

### Metrics (`metrics.py`)

**Attention Entropy (Baseline 1)**  
Computes `-sum(p * log p)` over each attention distribution (per head, per layer), then averages. High entropy = diffuse attention = uncertain = hallucinated.

**Logit Confidence (Baseline 2)**  
`1 - max(softmax(logits))` per token. High value = low confidence.

**Cosine Drift**  
`1 - cosine_similarity(h_t, h_{t-1})` averaged over layers 12–24. Measures how much the representation shifts between adjacent tokens. Large shift = representational instability.

**Mahalanobis Distance**  
μₗ and Σₗ are estimated from faithful (label=0) training samples only — this is critical for the leakage check. At test time, each token's hidden state is compared against the faithful manifold. Large distance = out-of-distribution = hallucinated.

**Logit Lens Divergence**  
Projects intermediate hidden states through the final LM head, computes KL(final || mid) for layers 6–23. High KL = representation still evolving late in the network = uncertain prediction.

**PCA Deviation**  
PCA fitted on faithful training hidden states. Reconstruction error of test token = variance not captured by the faithful subspace = hallucinated.

### Composite (`composite.py`)
- Each metric score is normalised to [0,1] via MinMaxScaler
- Composite = mean of all normalised scores
- Metrics added incrementally (Experiment 2 ablation)
- AUROC computed via `sklearn.metrics.roc_auc_score`
- 95% bootstrap CI with 1000 resamples
- F1 at best threshold, Spearman ρ, ECE also reported

### Layer localisation (E2)
- Cosine drift computed separately for each layer 1–24
- AUROC computed per layer
- Top-3 highest-AUROC layers identified as CIE layers
- Results plotted in `exp1_2_layer_profile.png`

---

## Leakage check
μₗ and Σₗ (Mahalanobis) and PCA components are fitted **only on the training split** before any test data is seen. Test samples are never used in fitting. The `split_dataset()` function uses a fixed seed for reproducibility.
