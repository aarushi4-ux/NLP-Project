import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# =========================
# 1. LOAD DATA (HALUEVAL)
# =========================
path = "/Users/sharo2/Downloads/projects/general_data.json"

raw = []
with open(path, "r") as f:
    for line in f:
        try:
            raw.append(json.loads(line))
        except:
            continue

print("Loaded:", len(raw))


print("\nRAW SAMPLE KEYS:")
print(raw[0].keys())

print("\nSAMPLE RECORD:")
print(raw[0])


# =========================
# 2. LABELS (ROBUST)
# =========================
def get_label(ex):
    v = ex.get("hallucination", 0)

    if isinstance(v, str):
        v = v.strip().lower()
        if v in ["yes", "true", "1"]:
            return 1
        if v in ["no", "false", "0"]:
            return 0
        return 0

    try:
        return int(v)
    except:
        return 0


# =========================
# 3. TEXT
# =========================
def get_text(ex):
    q = ex.get("user_query", "")
    a = ex.get("chatgpt_response", "")
    return (q + " " + a).strip()


data = [
    {"text": get_text(x), "label": get_label(x)}
    for x in raw
    if get_text(x)
]

print("✔ usable:", len(data))

labels_check = [d["label"] for d in data]
print("Label distribution:", np.unique(labels_check, return_counts=True))


# =========================
# 4. EXTRACTOR
# =========================
from composite import HiddenStateExtractor

extractor = HiddenStateExtractor(device="mps")


def extract_hidden_states(text):
    out = extractor.extract(text)
    hs = out["hidden_states"] if isinstance(out, dict) else out.hidden_states
    hs = np.asarray(hs)

    # trim embedding layer if needed
    if hs.shape[0] > 24:
        hs = hs[:24]

    return hs


# =========================
# 5. LOAD CHECKPOINT
# =========================
obj = torch.load("exp1_2_results.pt", map_location="cpu", weights_only=False)

maha = obj["mahalanobis"]
pca = obj["pca"]

print("MAHA layers:", len(maha["mu"]))
print("PCA layers:", len(pca["components"]))


# =========================
# 6. OPTIONAL: RAGTRUTH BASELINE (FROM .PT)
# =========================
ragtruth_auc = obj.get("ragtruth_auc", None)  # <-- IMPORTANT ADD

if ragtruth_auc is not None:
    print("\nRAGTruth (from checkpoint):", ragtruth_auc)
else:
    print("\n⚠️ No RAGTruth baseline found in checkpoint")


# =========================
# 7. METRICS
# =========================
def maha_score(hs):
    scores = []
    for layer in sorted(maha["mu"].keys(), key=int):
        layer = int(layer)
        if layer >= hs.shape[0]:
            continue

        mu = np.asarray(maha["mu"][layer])
        inv_cov = np.asarray(maha["inv_cov"][layer])

        h = hs[layer].mean(axis=0)
        diff = h - mu

        scores.append(float(diff @ inv_cov @ diff))

    return np.mean(scores) if scores else 0.0


def pca_score(hs):
    scores = []
    for layer in sorted(pca["components"].keys(), key=int):
        layer = int(layer)
        if layer >= hs.shape[0]:
            continue

        comps = np.asarray(pca["components"][layer])
        mean = np.asarray(pca["mean"][layer])

        h = hs[layer].mean(axis=0)
        proj = (h - mean) @ comps.T

        scores.append(float(np.sum(proj ** 2)))

    return np.mean(scores) if scores else 0.0


def cosine(hs):
    return float(np.mean(np.linalg.norm(hs, axis=-1)))


def logit(hs):
    return float(np.mean(np.var(hs, axis=-1)))


def cie(hs):
    return float(np.mean(np.abs(hs)))


# =========================
# 8. LOOP
# =========================
y_true = []
scores = {"cosine": [], "mahal": [], "logit": [], "pca": [], "cie": []}

for ex in tqdm(data):
    try:
        hs = extract_hidden_states(ex["text"])

        y_true.append(ex["label"])

        scores["cosine"].append(cosine(hs))
        scores["mahal"].append(maha_score(hs))
        scores["logit"].append(logit(hs))
        scores["pca"].append(pca_score(hs))
        scores["cie"].append(cie(hs))

    except Exception as e:
        print("❌ ERROR:", repr(e))
        continue


# =========================
# 9. AUROC SAFE
# =========================
y_true = np.array(y_true)
scores = {k: np.array(v) for k, v in scores.items()}


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)

    if len(np.unique(y)) < 2:
        return float("nan")
    if len(np.unique(s)) < 2:
        return float("nan")

    return roc_auc_score(y, s)


print("\n=== HALUEVAL RESULTS ===\n")

results = {}
for k in scores:
    auc = safe_auc(y_true, scores[k])
    results[k] = auc
    print(f"{k:10s} | AUROC: {auc:.4f}")


# =========================
# 10. COMPOSITE
# =========================
def composite(S):
    return (
        0.25 * S["cosine"] +
        0.25 * S["mahal"] +
        0.20 * S["logit"] +
        0.15 * S["pca"] +
        0.15 * S["cie"]
    )


comp_auc = safe_auc(y_true, composite(scores))
print("\nCOMPOSITE AUROC:", comp_auc)


# =========================
# 11. EXP5 SUMMARY (RAG vs HALUEVAL DROP)
# =========================
print("\n=== EXP 5 CROSS-DOMAIN SUMMARY ===")

if ragtruth_auc is not None:
    print(f"RAGTruth baseline : {ragtruth_auc:.4f}")
    print(f"HaluEval composite: {comp_auc:.4f}")
    print(f"Drop             : {ragtruth_auc - comp_auc:.4f}")
else:
    print("RAGTruth baseline missing in .pt — cannot compute drop")