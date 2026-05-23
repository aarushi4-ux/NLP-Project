# Hallucination Detection in RAG Systems via Hidden State Analysis

A representation-based hallucination detection framework for RAG (Retrieval-Augmented Generation) outputs. This project uses GPT-2-medium's internal hidden states — attention patterns, cosine drift, Mahalanobis distance, PCA deviation, logit lens divergence, and CIE layer localisation — to detect hallucinated tokens without any fine-tuning.

---

## Datasets

- **RAGTruth** — [github.com/ParakweetLab/RAGTruth](https://github.com/ParakweetLab/RAGTruth): a benchmark of LLM responses to RAG prompts with fine-grained hallucination span annotations. Used for training, evaluation, and all core experiments (Exp 1–4, 6–8).
- **HaluEval** — [github.com/RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval): a large-scale hallucination evaluation dataset covering QA, summarisation, and dialogue. Used in Exp 5 for cross-domain zero-shot transfer.

---

## Project Structure

```
.
├── Exp1_2/
│   ├── data_loader.py        # RAGTruth loader → RAGSample dataclass
│   ├── extractor.py          # GPT-2-medium hidden state extractor
│   ├── metrics.py            # All 7 representation metrics
│   ├── composite.py          # Exp 1 & 2 pipeline; saves exp1_2_results.pt
│   ├── plot_final.py         # Layer-profile and AUROC bar chart plots
│   └── requirements.txt
│
├── Exp3/
│   ├── data_loader.py        # Sample loader using exp1_2_results.pt test split
│   ├── pair_selector.py      # Builds (faithful, hallucinated) pairs by source_id
│   ├── patch_runner.py       # Activation patching + CIE scoring
│   └── run_exp3.py           # Entry point; saves exp3_results.csv + e4_layer_config.json
│
├── Exp4/
│   ├── preprocessing.py      # Character → token span alignment; saves e4_processed.json
│   └── run_exp4.py           # Temporal precedence analysis (t-3 … t+1 window)
│
├── Exp5/
│   └── haluevaltest.py       # Zero-shot cross-domain evaluation on HaluEval
│
├── Exp6/  →  run_exp6.py     # FFN vs attention AUROC decomposition by layer range
├── Exp7/  →  run_exp7.py     # Error analysis; exports FP/FN CSVs + report cases
├── Exp8/  →  run_exp8.py     # SOTA gap comparison (ReDeEP, LUMINA baselines)
│
└── demo.py                   # Live single-sample hallucination scorer
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Transformers ≥ 4.38, scikit-learn, scipy, numpy, matplotlib.

GPT-2-medium (~1.5 GB) is downloaded automatically on first run via HuggingFace.

---

## Reproducing Results

### Step 1 — Exp 1 & 2: Core metrics + composite (required first)

This is the main pipeline. It fits all metrics on the RAGTruth train split, scores the test split, and saves `exp1_2_results.pt` — which all downstream experiments load.

```bash
cd Exp1_2
python composite.py --response /path/to/response.jsonl --source /path/to/source.jsonl
python plot_final.py   # generates layer-profile and AUROC bar charts
```

### Step 2 — Exp 3: Activation patching & CIE

```bash
cd Exp3
python run_exp3.py --csv /path/to/metrics_per_example.csv --pt ../outputs/exp1_2_results.pt
```

### Step 3 — Exp 4: Temporal precedence

```bash
cd Exp4
python preprocessing.py   # produces e4_processed.json
python run_exp4.py
```

### Step 4 — Exp 5: Cross-domain transfer (HaluEval)

```bash
cd Exp5
# Update the path to general_data.json inside haluevaltest.py, then:
python haluevaltest.py
```

### Steps 5–7 — Exp 6, 7, 8

Each script loads `exp1_2_results.pt` directly and runs standalone.

```bash
python run_exp6.py
python run_exp7.py
python run_exp8.py
```

### Live Demo

```bash
python demo.py \
  --context "Your retrieved passage" \
  --question "Your question" \
  --response "The model's answer" \
  --pt ../outputs/exp1_2_results.pt
```

Outputs a token-level table of all 7 metric scores and an overall hallucination likelihood flag (LOW / WEAK / UNCERTAIN / HIGH).

---

## Key Output File

`exp1_2_results.pt` is the central artifact produced by Exp 1 & 2. It contains fitted metric parameters (Mahalanobis μ/Σ, PCA components, CIE top layers), per-sample and per-token scores, normalisation constants, and layer profiles. All downstream experiments (3–8) and the demo load from this file — **do not re-fit on HaluEval or any other dataset**.
