# Demo — Live Token-Level Hallucination Detection

## What This Does

`demo.py` is a self-contained interactive demo that takes a single (context, question, response) triple, runs it through GPT-2 Medium, and produces a **token-level hallucination report** using all seven metrics from Exp 1/2.

It is designed to let evaluators or teammates test the pipeline on arbitrary inputs without running the full experiment. Frozen parameters from `exp1_2_results.pt` are loaded directly — no re-fitting happens.

---

## Files

| File | Role |
|---|---|
| `demo.py` | Entry point — builds the prompt, extracts hidden states, scores every response token, flags suspicious tokens, and prints a summary |

Internally calls:
- `extractor.py` → `HiddenStateExtractor` (GPT-2 Medium forward pass)
- `composite.py` → `make_cpu_lm_head` (CPU copy of LM head for logit lens)
- `metrics.py` → all seven metric functions / classes

---

## Inputs

### Via command-line arguments

```bash
python demo.py \
  --context  "Your source passage here" \
  --question "Question being answered" \
  --response "Model's response to evaluate" \
  --pt       "../outputs/exp1_2_results.pt"
```

### Default inputs (used when no arguments are passed)

```python
DEFAULT_CONTEXT  = "The Eiffel Tower is located in Paris, France. It was constructed
                    in 1889 as the entrance arch for the 1889 World's Fair. It stands
                    330 metres tall."
DEFAULT_QUESTION = "Where is the Eiffel Tower and when was it built?"
DEFAULT_RESPONSE = "The Eiffel Tower was built in 1889 and is located in Paris, France.
                    It stands 330 metres tall."
```

| Argument | Default | Description |
|---|---|---|
| `--context` | Eiffel Tower passage | Source document / retrieved context |
| `--question` | Eiffel Tower question | Question the response is answering |
| `--response` | Eiffel Tower answer | The model response to evaluate for hallucination |
| `--pt` | `../outputs/exp1_2_results.pt` | Path to frozen Exp 1/2 checkpoint |

---

## Outputs

All output is printed to the terminal in three sections.

### 1. Token-level metric table

A row per response token showing raw scores for all seven metrics:

```
Token                Attn Entropy    Logit Conf    Cosine Drift    Mahalanobis    Logit Lens Div    PCA Deviation    CIE top-3
ĠEiffel                    2.3412        0.9871          0.0023         14.2341           8.3421           23.1234       0.0041
ĠTower                     2.2981        0.9812          0.0019         13.9821           8.1234           22.8123       0.0038
...
```

### 2. Token-level hallucination flags

Tokens where the **weighted composite score** exceeds a threshold are flagged with a severity level:

| Level | Composite threshold |
|---|---|
| `HIGH` | > 0.60 |
| `UNCERTAIN` | > 0.40 |
| `WEAK` | > 0.25 |

```
TOKEN-LEVEL HALLUCINATION FLAGS
Ġin1987              [HIGH     ]  composite: 0.7341
Ġkilometres          [UNCERTAIN]  composite: 0.4812
```

### 3. Overall verdict

```
Flagged: 2 tokens  (1 HIGH, 1 UNCERTAIN, 0 WEAK)
Overall hallucination likelihood: HIGH
```

Overall verdict logic:

| Condition | Verdict |
|---|---|
| ≥ 1 HIGH token | `HIGH` |
| ≥ 2 UNCERTAIN tokens | `UNCERTAIN` |
| 1 UNCERTAIN + ≥ 2 WEAK | `UNCERTAIN` |
| ≥ 3 flagged tokens (any level) | `UNCERTAIN` |
| Otherwise | `LOW` |

---

## How Scores Are Computed

The demo loads frozen parameters from the `.pt` checkpoint and reuses them exactly as in Exp 1/2 — no re-fitting:

- **Mahalanobis**: `μₗ` and `Σₗ⁻¹` loaded from `mahalanobis.mu` / `mahalanobis.inv_cov`
- **PCA**: components and per-layer mean loaded from `pca.components` / `pca.mean`
- **CIE**: full `CIETopLayersMetric` object with `top_layers` already set, loaded from `cie_full`
- **Normalisation**: `normalization.min` / `normalization.max` applied before weighting
- **Weights**: AUROC-proportional weights from `weights` (clipped below 0.5, then softmax-normalised)

The composite per token is:

```
normed  = clip((raw − norm_min) / (norm_max − norm_min + ε), 0, 1)
composite = (normed × weights).sum()
```

---

## How to Run

```bash
# With defaults (Eiffel Tower example)
python demo.py

# With custom input
python demo.py \
  --context  "Marie Curie was born in Warsaw in 1867." \
  --question "When and where was Marie Curie born?" \
  --response "Marie Curie was born in Paris in 1867." \
  --pt       "../outputs/exp1_2_results.pt"
```

**Device:** Automatically selects MPS → CUDA → CPU. GPT-2 Medium fits comfortably in CPU RAM (~1.5 GB); expect ~5–15 seconds per run on CPU.

**Dependency:** `exp1_2_results.pt` must exist (produced by running `composite.py`). If it is missing, the script will raise a `FileNotFoundError` at the `torch.load` call.
