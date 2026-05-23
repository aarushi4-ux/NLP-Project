# Experiment 3 — Causal Intervention Effect (CIE) via Activation Patching

## What This Experiment Measures

Experiment 3 identifies **which transformer layers causally drive hallucination** in GPT-2 Medium by running activation patching between matched faithful/hallucinated sample pairs.

For each (faithful, hallucinated) pair, the experiment patches attention or FFN activations from one sample into the forward pass of the other — in both directions:

- **f→h**: patch faithful activations into the hallucinated forward pass
- **h→f**: patch hallucinated activations into the faithful forward pass

The **Causal Intervention Effect (CIE)** for a given layer is measured as the total variation distance between the original and patched output distributions (`|P_base − P_patched|`). A large CIE indicates that layer causally encodes information relevant to hallucination.

Results are summarised by layer band (early / mid / late) and component (attention vs FFN), and the top-5 most causally important layers per component are saved for Experiment 4.

---

## Files

| File | Role |
|---|---|
| `run_exp3.py` | Entry point — orchestrates loading, pairing, patching, and saving results |
| `pair_selector.py` | Builds matched (faithful, hallucinated) pairs grouped by `source_id` |
| `patch_runner.py` | Runs the patching loop; extracts attention/FFN activations and computes CIE |

---

## Inputs

| Input | Description |
|---|---|
| `--csv` | `metrics_per_example.csv` — per-example metrics output from Exp 1/2 |
| `--pt` | `exp1_2_results.pt` — frozen checkpoint from Exp 1/2, used for test split IDs and labels |
| GPT-2 Medium | Downloaded automatically via HuggingFace on first run |

The `data_loader.py` in this experiment reads the CSV and `.pt` file to reconstruct the test split, filtering to only those `example_id`s present in `test_sample_ids` from the checkpoint.

---

## Outputs

| Output | Description |
|---|---|
| `exp3_results.csv` | One row per (pair, direction, layer, component) with CIE score |
| `activations/` | Directory of saved `.pt` activation tensors per pair (faithful/halluc × attn/FFN/residual) |
| `../outputs/e4_layer_config.json` | Top-5 attention and FFN layers ranked by mean CIE; consumed by Exp 4 |

### `exp3_results.csv` columns

`pair_idx`, `pair_id`, `direction` (f_to_h / h_to_f), `layer`, `component` (attn / ffn), `cie`, `n_layers`

### `e4_layer_config.json` structure

```json
{
  "top_attn_layers": [14, 13, 9, 11, 2],
  "top_ffn_layers":  [0, 23, 22, 21, 20],
  "top_layers_combined": [0, 2, 9, 11, 13, 14, 20, 21, 22, 23]
}
```

---

## How to Run

```bash
python run_exp3.py \
  --csv  ../outputs/metrics_per_example.csv \
  --pt   ../outputs/exp1_2_results.pt \
  --out  ../outputs/exp3_results.csv \
  --max_pairs  147 \
  --max_length 128 \
  --cie_mode   last
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | required | Path to per-example metrics CSV |
| `--pt` | required | Path to `exp1_2_results.pt` |
| `--out` | `../outputs/exp3_results.csv` | Output CSV path |
| `--max_pairs` | `147` | Max number of (faithful, hallucinated) pairs to patch |
| `--max_length` | `128` | Token truncation length for GPT-2 input |
| `--cie_mode` | `last` | Aggregation mode for CIE: `last`, `mean`, or `max` |

**Runtime note:** With 147 pairs × 24 layers × 2 directions × 2 components = ~14,000 patching steps. Expect ~20–40 min on CPU, ~5–10 min with a GPU.

---

## Pair Construction (`pair_selector.py`)

Pairs are built by grouping samples on `source_id` (same source document) and matching each faithful sample to a randomly chosen hallucinated sample from the same group. After exhausting aligned pairs, random cross-group pairs are added as fallback until `max_pairs` is reached.

---

## Patching Mechanics (`patch_runner.py`)

For each layer and component:

1. Extract full attention and FFN activations from both the faithful and hallucinated forward passes using `register_forward_hook`.
2. Run a second forward pass on the target sequence with a hook that **replaces** the target layer's output with the source's activation.
3. Compute CIE as `|softmax(base_logits) − softmax(patched_logits)|` summed over the vocabulary and averaged over token positions (mode controlled by `--cie_mode`).

Activations are also saved to `activations/` as stacked tensors of shape `(n_layers, seq_len, hidden_dim)` for downstream use in Exp 4 temporal analysis.
