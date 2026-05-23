import json
import torch
import numpy as np
import csv

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None

# ─────────────────────────────────────────
# 1. Load .pt file
# ─────────────────────────────────────────
data = torch.load("exp1_2_results.pt", map_location="cpu", weights_only=False)

test_labels = data["test_labels"]
test_ids    = data["test_sample_ids"]

# Convert labels to numpy
if hasattr(test_labels, "cpu"):
    test_labels = test_labels.cpu().numpy()
else:
    test_labels = np.array(test_labels)

print("Labels shape :", test_labels.shape)
print("Num test IDs :", len(test_ids))

# ─────────────────────────────────────────
# 2. Build composite score
#    (equal-weight average of all metrics)
# ─────────────────────────────────────────
def collapse(score_list):
    """Per-token list/tensor → one float per sample (mean)."""
    out = []
    for s in score_list:
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        flat = np.array(s, dtype=float).reshape(-1)
        out.append(float(np.mean(flat)) if len(flat) > 0 else 0.0)
    return np.array(out, dtype=float)

per_token        = data["per_token_scores"]
available_metrics = list(per_token.keys())
print("Available metrics:", available_metrics)

tokenizer = None
if GPT2TokenizerFast is not None:
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    except Exception as e:
        print(f"[E7] GPT-2 tokenizer unavailable, using whitespace fallback: {e}")

composite = np.zeros(len(test_labels), dtype=float)
for metric in available_metrics:
    composite += collapse(per_token[metric])
composite /= len(available_metrics)          # equal-weight average

print(f"Composite score — min: {composite.min():.4f}  "
      f"max: {composite.max():.4f}  mean: {composite.mean():.4f}")

# ─────────────────────────────────────────
# 3. Threshold
#    Set TAU to the mean so ~50 % are pred=1
# ─────────────────────────────────────────
TAU   = float(composite.mean())              # data-driven threshold
preds = (composite >= TAU).astype(int)

print(f"\nThreshold tau = {TAU:.4f}")
print(f"Pred=1 (hallucinated): {preds.sum()}  |  "
      f"Pred=0 (faithful): {(preds == 0).sum()}")

# ─────────────────────────────────────────
# 4. Load raw dataset  (JSONL or JSON array)
# ─────────────────────────────────────────
def as_flat(arr):
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    return np.array(arr, dtype=float).reshape(-1)


def token_level_metric_matrix(sample_idx):
    metric_arrays = [as_flat(per_token[m][sample_idx]) for m in available_metrics]
    min_len = min((len(a) for a in metric_arrays), default=0)
    if min_len == 0:
        return np.zeros((0, len(available_metrics))), np.zeros(0)
    matrix = np.column_stack([a[:min_len] for a in metric_arrays])
    return matrix, matrix.mean(axis=1)


def response_tokens(response_text, expected_len):
    if tokenizer is not None and response_text:
        toks = tokenizer.tokenize(response_text)
        toks = [t.replace("Ġ", "").replace("Ċ", "\\n") or t for t in toks]
    else:
        toks = response_text.split()

    if not toks:
        toks = ["<empty>"]
    if expected_len <= 0:
        return toks
    if len(toks) < expected_len:
        toks = toks + ["<pad>"] * (expected_len - len(toks))
    return toks[:expected_len]


def label_name(label):
    return "hallucinated" if int(label) == 1 else "faithful"


def explain_failure(row, dominant_metric, token_score):
    is_fp = row["true_label"] == 0 and row["pred_label"] == 1
    metric = dominant_metric.lower()

    if is_fp:
        if "mahal" in metric or "pca" in metric:
            return "The token looked out-of-distribution in hidden-state space, so a faithful rare span was mistaken for hallucination."
        if "logit" in metric:
            return "The model showed lexical uncertainty, so a correct but low-confidence answer token pushed the composite above threshold."
        if "attention" in metric:
            return "Diffuse attention around a correct token inflated the drift signal even though the answer was faithful."
        return "A local representation jump in a correct answer inflated drift and made faithful content resemble hallucination."

    if token_score < TAU:
        return "The hallucinated token remained fluent and representation-stable, so the drift signal stayed below the decision threshold."
    return "The token showed some drift, but averaging with weaker metrics diluted the signal below the sample-level decision boundary."


def enrich_failure_case(row):
    idx = int(row["row_idx"])
    metric_matrix, token_comp = token_level_metric_matrix(idx)
    toks = response_tokens(row["response_text"], 0)
    valid_len = min(len(toks), len(token_comp))

    if len(token_comp) == 0:
        token_idx = 0
        token_score = float(row["score"])
        dominant_metric = available_metrics[0] if available_metrics else "composite"
    else:
        search_scores = token_comp[:valid_len] if valid_len > 0 else token_comp
        token_idx = int(np.argmax(search_scores))
        token_score = float(token_comp[token_idx])
        dominant_metric = available_metrics[int(np.argmax(metric_matrix[token_idx]))]

    token = toks[token_idx] if token_idx < len(toks) else "<unk>"
    explanation = explain_failure(row, dominant_metric, token_score)

    return {
        **row,
        "token_index": token_idx,
        "misclassified_token": token,
        "token_composite": round(token_score, 6),
        "tau": round(TAU, 6),
        "dominant_metric": dominant_metric,
        "mechanistic_explanation": explanation,
        "report_sentence": (
            f"Token `{token}` was misclassified; ground truth={label_name(row['true_label'])}, "
            f"composite={float(row['score']):.4f}, tau={TAU:.4f}. {explanation}"
        ),
    }


raw       = []
data_path = "../data/qa_data.json"

with open(data_path, "r", encoding="utf-8") as f:
    content = f.read().strip()

try:                                         # JSON array
    raw = json.loads(content)
    if not isinstance(raw, list):
        raw = [raw]
    print(f"\nLoaded JSON array: {len(raw)} examples")
except json.JSONDecodeError:                 # JSONL
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw.append(json.loads(line))
        except Exception as e:
            print(f"Skipping bad JSON line {line_num}: {e}")
    print(f"\nLoaded JSONL: {len(raw)} examples")

if raw:
    print("Sample keys:", list(raw[0].keys()))

# ─────────────────────────────────────────
# 5. Map sample_id → example
#    Falls back to positional index when no
#    explicit id field exists in the dataset.
# ─────────────────────────────────────────
id_to_example = {}
for idx, ex in enumerate(raw):
    ex_id = (ex.get("id")        or ex.get("sample_id") or
             ex.get("uid")       or ex.get("example_id"))
    key = str(ex_id) if ex_id is not None else str(idx)
    id_to_example[key] = ex

print(f"Mapped {len(id_to_example)} IDs from dataset")

# ─────────────────────────────────────────
# 6. Build rows
# ─────────────────────────────────────────
rows = []
for row_idx, (sid, y, s, p) in enumerate(zip(test_ids, test_labels, composite, preds)):
    ex = id_to_example.get(str(sid), {})

    question_text = (ex.get("question") or ex.get("prompt") or
                     ex.get("query")    or ex.get("input")  or "")

    # Pick the response that matches the true label
    if int(y) == 1:                          # truly hallucinated
        response_text = (ex.get("hallucinated_answer") or
                         ex.get("response")            or
                         ex.get("answer")              or
                         ex.get("output")              or "")
    else:                                    # truly faithful
        response_text = (ex.get("right_answer")  or
                         ex.get("response")      or
                         ex.get("answer")        or
                         ex.get("output")        or "")

    rows.append({
        "row_idx":       row_idx,
        "sample_id":     sid,
        "true_label":    int(y),
        "score":         round(float(s), 6),
        "pred_label":    int(p),
        "question":      question_text,
        "response_text": response_text,
    })

FIELDNAMES = ["row_idx", "sample_id", "true_label", "score", "pred_label",
              "question", "response_text"]

# ─────────────────────────────────────────
# 7. Save ALL cases
# ─────────────────────────────────────────
with open("exp7_cases.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
print("\nSaved exp7_cases.csv  (all samples)")

# ─────────────────────────────────────────
# 8. Split into FP and FN
#
#   False Positive : true=0 (faithful)     pred=1 (wrongly flagged hallucinated)
#   False Negative : true=1 (hallucinated) pred=0 (missed — predicted faithful)
# ─────────────────────────────────────────
false_positives = sorted(
    [r for r in rows if r["true_label"] == 0 and r["pred_label"] == 1],
    key=lambda x: -x["score"]    # highest score first (worst FPs)
)
false_negatives = sorted(
    [r for r in rows if r["true_label"] == 1 and r["pred_label"] == 0],
    key=lambda x:  x["score"]    # lowest score first (most missed FNs)
)

with open("exp7_false_positives.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(false_positives)
print(f"Saved exp7_false_positives.csv  ({len(false_positives)} cases)")

with open("exp7_false_negatives.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(false_negatives)
print(f"Saved exp7_false_negatives.csv  ({len(false_negatives)} cases)")

# ─────────────────────────────────────────
# 9. Print top cases to terminal
# ─────────────────────────────────────────
def print_cases(cases, label, n=3):
    print(f"\n{'='*70}")
    print(f"TOP {n} {label}")
    print(f"{'='*70}")
    if not cases:
        print("  (none found)")
        return
    for r in cases[:n]:
        print(f"\nID    : {r['sample_id']}")
        print(f"True  : {r['true_label']}  |  Pred : {r['pred_label']}  |  Score : {r['score']}")
        print(f"Q     : {r['question'][:300]}")
        print(f"Resp  : {r['response_text'][:400]}")

print_cases(false_positives, "FALSE POSITIVES  (faithful -> wrongly flagged as hallucinated)")
print_cases(false_negatives, "FALSE NEGATIVES  (hallucinated -> missed, predicted faithful)")

# ─────────────────────────────────────────
# 10. Confusion matrix summary
# ─────────────────────────────────────────
usable_fp = [r for r in false_positives if r["response_text"].strip()]
usable_fn = [r for r in false_negatives if r["response_text"].strip()]
report_seed = usable_fp[:2] + usable_fn[:1]
if len(report_seed) < 3:
    remaining = [
        r for r in (usable_fp + usable_fn + false_positives + false_negatives)
        if r not in report_seed
    ]
    report_seed.extend(remaining[: 3 - len(report_seed)])

report_cases = [enrich_failure_case(r) for r in report_seed[:3]]
REPORT_FIELDNAMES = FIELDNAMES + [
    "token_index",
    "misclassified_token",
    "token_composite",
    "tau",
    "dominant_metric",
    "mechanistic_explanation",
    "report_sentence",
]

with open("exp7_report_cases.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=REPORT_FIELDNAMES)
    writer.writeheader()
    writer.writerows(report_cases)

print(f"\n{'='*70}")
print("E7 REPORT-READY FAILURE CASES")
print(f"{'='*70}")
for i, r in enumerate(report_cases, start=1):
    print(f"\nCase {i}: {r['report_sentence']}")
    print(f"  token_index={r['token_index']} token_composite={r['token_composite']} "
          f"dominant_metric={r['dominant_metric']}")
print("\nSaved exp7_report_cases.csv")

tp = sum(1 for r in rows if r["true_label"] == 1 and r["pred_label"] == 1)
tn = sum(1 for r in rows if r["true_label"] == 0 and r["pred_label"] == 0)
fp = len(false_positives)
fn = len(false_negatives)
total = tp + tn + fp + fn

print(f"\n{'='*70}")
print("CONFUSION MATRIX SUMMARY")
print(f"  TP (correctly flagged hallucinations) : {tp}")
print(f"  TN (correctly passed faithful)        : {tn}")
print(f"  FP (faithful wrongly flagged)         : {fp}")
print(f"  FN (hallucinations missed)            : {fn}")
if total > 0:
    print(f"  Accuracy  : {(tp + tn) / total:.4f}")
if (tp + fp) > 0:
    print(f"  Precision : {tp / (tp + fp):.4f}")
if (tp + fn) > 0:
    print(f"  Recall    : {tp / (tp + fn):.4f}")
print(f"\nThreshold used: tau = {TAU:.4f}")
print("Done.")
