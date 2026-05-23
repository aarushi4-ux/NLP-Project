import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr


def as_array(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


def safe_auc(labels, scores):
    scores = as_array(scores)
    if len(scores) == 0 or np.std(scores) <= 1e-12:
        return 0.5
    return float(roc_auc_score(labels, scores))


def range_scores(profile, layers):
    values = [as_array(profile[l]) for l in layers if l in profile]
    if not values:
        return np.array([], dtype=float)
    min_len = min(len(v) for v in values)
    return np.column_stack([v[:min_len] for v in values]).mean(axis=1)


# -----------------------------
# Load saved E1/E2 component scores
# -----------------------------
data = torch.load("exp1_2_results.pt", map_location="cpu", weights_only=False)

labels = as_array(data["test_labels"]).astype(int)
attn_scores = data["attn_layer_scores"]
ffn_scores = data["ffn_layer_scores"]
attn_profile = data["attn_layer_profile"]
ffn_profile = data["ffn_layer_profile"]
layer_scores_raw = data["layer_scores_raw"]

# GPT-2 medium has 24 transformer layers.
ranges = {
    "Early (1-25%)": list(range(1, 7)),
    "Mid (26-75%)": list(range(7, 19)),
    "Late (76-100%)": list(range(19, 25)),
    "Mid-to-late": list(range(7, 25)),
}


print("\n===== E6: FFN vs Attention Decomposition =====\n")
print(
    f"{'Layer range':<16} {'FFN AUROC':>10} {'Attn AUROC':>11} "
    f"{'FFN |AUC-.5|':>13} {'Attn |AUC-.5|':>14} {'Winner':>10}"
)
print("-" * 82)

range_results = {}

for name, layers in ranges.items():
    ffn_range = range_scores(ffn_scores, layers)
    attn_range = range_scores(attn_scores, layers)

    n = min(len(labels), len(ffn_range), len(attn_range))
    y = labels[:n]
    ffn_range = ffn_range[:n]
    attn_range = attn_range[:n]

    ffn_auc = safe_auc(y, ffn_range)
    attn_auc = safe_auc(y, attn_range)
    ffn_strength = abs(ffn_auc - 0.5)
    attn_strength = abs(attn_auc - 0.5)

    if ffn_strength > attn_strength:
        winner = "FFN"
    elif attn_strength > ffn_strength:
        winner = "Attention"
    else:
        winner = "Tie"

    range_results[name] = {
        "ffn_auc": ffn_auc,
        "attn_auc": attn_auc,
        "ffn_strength": ffn_strength,
        "attn_strength": attn_strength,
        "winner": winner,
    }

    print(
        f"{name:<16} {ffn_auc:>10.4f} {attn_auc:>11.4f} "
        f"{ffn_strength:>13.4f} {attn_strength:>14.4f} {winner:>10}"
    )


print("\n===== Layer-wise AUROC Profiles =====\n")
print(f"{'Layer':<8} {'FFN AUROC':>10} {'Attn AUROC':>11} {'Winner':>10}")
print("-" * 45)

for layer in sorted(attn_profile.keys()):
    ffn_auc = float(ffn_profile[layer])
    attn_auc = float(attn_profile[layer])
    ffn_strength = abs(ffn_auc - 0.5)
    attn_strength = abs(attn_auc - 0.5)

    if ffn_strength > attn_strength:
        winner = "FFN"
    elif attn_strength > ffn_strength:
        winner = "Attention"
    else:
        winner = "Tie"

    print(f"{layer:<8} {ffn_auc:>10.4f} {attn_auc:>11.4f} {winner:>10}")

print("\n===== Point-Biserial Layer Profile (Cosine Drift vs Label) =====\n")
print(f"{'Layer':<8} {'r':>10} {'p-value':>12} {'abs(r)':>10}")
print("-" * 45)

pb_results = {}
for layer in sorted(layer_scores_raw.keys()):
    scores = as_array(layer_scores_raw[layer])
    n = min(len(labels), len(scores))
    y = labels[:n]
    s = scores[:n]

    if n < 2 or len(np.unique(y)) < 2 or np.std(s) <= 1e-12:
        r, p = np.nan, np.nan
    else:
        r, p = pointbiserialr(y, s)
        r, p = float(r), float(p)

    pb_results[layer] = {"r": r, "p": p, "abs_r": abs(r) if not np.isnan(r) else np.nan}
    print(f"{layer:<8} {r:>10.4f} {p:>12.4g} {pb_results[layer]['abs_r']:>10.4f}")

valid_pb = {l: v for l, v in pb_results.items() if not np.isnan(v["abs_r"])}
if valid_pb:
    best_layer = max(valid_pb, key=lambda l: valid_pb[l]["abs_r"])
    best = valid_pb[best_layer]
    print(
        f"\nStrongest point-biserial layer: {best_layer} "
        f"(r={best['r']:.4f}, p={best['p']:.4g}, |r|={best['abs_r']:.4f})"
    )


early = range_results["Early (1-25%)"]
mid = range_results["Mid (26-75%)"]
late = range_results["Late (76-100%)"]
mid_late = range_results["Mid-to-late"]

print("\n===== Interpretation =====\n")
print(
    f"Early winner: {early['winner']} "
    f"(FFN strength={early['ffn_strength']:.4f}, "
    f"Attention strength={early['attn_strength']:.4f})"
)
print(
    f"Mid winner: {mid['winner']} "
    f"(FFN strength={mid['ffn_strength']:.4f}, "
    f"Attention strength={mid['attn_strength']:.4f})"
)
print(
    f"Late winner: {late['winner']} "
    f"(FFN strength={late['ffn_strength']:.4f}, "
    f"Attention strength={late['attn_strength']:.4f})"
)
print(
    f"Mid-to-late winner: {mid_late['winner']} "
    f"(FFN strength={mid_late['ffn_strength']:.4f}, "
    f"Attention strength={mid_late['attn_strength']:.4f})"
)

if mid_late["winner"] == "FFN":
    print("\nFinding: The E6 signal localizes mainly to mid-to-late FFN components.")
else:
    print("\nFinding: The E6 signal does not mainly localize to mid-to-late FFN; attention remains stronger overall.")
