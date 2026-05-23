import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path

# --------------------------------------------------
# Load
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "outputs"
PT_FILE = OUTPUT_DIR / "exp1_2_results.pt"

print("Loading:", PT_FILE)
results = torch.load(PT_FILE, map_location="cpu", weights_only=False)

test_labels = np.array(results["test_labels"])
layer_scores_raw = results["layer_scores_raw"]
metric_scores = results["metric_scores"]

N_LAYERS = max(layer_scores_raw.keys()) + 1

# --------------------------------------------------
# Compute AUROC per layer (for layer-profile metrics)
# --------------------------------------------------
def compute_layer_auroc(layer_scores, labels):
    out = {}
    for l, scores in layer_scores.items():
        scores = np.array(scores, dtype=float)
        mask = np.isfinite(scores)
        s = scores[mask]
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        try:
            auc = roc_auc_score(y, s)
            out[l] = max(auc, 1 - auc)
        except:
            continue
    return out

def compute_scalar_auroc(scores, labels):
    """For flat per-sample score arrays (non-layer metrics)."""
    scores = np.array(scores, dtype=float)
    mask = np.isfinite(scores)
    s = scores[mask]
    y = labels[mask]
    if len(np.unique(y)) < 2:
        return None
    try:
        auc = roc_auc_score(y, s)
        return max(auc, 1 - auc)
    except:
        return None

# --------------------------------------------------
# Figure 1: Layer-Profile Plot (Cosine Drift + Attn)
# --------------------------------------------------
layer_auroc = compute_layer_auroc(layer_scores_raw, test_labels)
layers = np.array(sorted(layer_auroc.keys()))
aurocs = np.array([layer_auroc[l] for l in layers])
best_layer = layers[np.argmax(aurocs)]
best_score = np.max(aurocs)

attn_auroc = {}
if "attn_layer_scores" in results:
    attn_auroc = compute_layer_auroc(results["attn_layer_scores"], test_labels)
    attn_layers = np.array(sorted(attn_auroc.keys()))
    attn_aurocs = np.array([attn_auroc[l] for l in attn_layers])

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(layers, aurocs, marker="o", linewidth=2, label="Cosine Drift")

if attn_auroc:
    ax.plot(attn_layers, attn_aurocs, marker="s", linewidth=2,
            linestyle="--", label="Attention Entropy (layer)")

# shade regions
colors = ["#4477AA", "#66CCEE", "#228833"]
for (lo, hi, lbl, c) in [
    (0,              N_LAYERS*0.33, "Early", colors[0]),
    (N_LAYERS*0.33,  N_LAYERS*0.66, "Mid",   colors[1]),
    (N_LAYERS*0.66,  N_LAYERS,      "Late",  colors[2]),
]:
    ax.axvspan(lo, hi, alpha=0.10, color=c, label=lbl)

ax.axhline(0.5, linestyle="--", linewidth=1, color="grey", label="Chance")
ax.scatter(best_layer, best_score, s=120, zorder=5)
ax.annotate(
    f"Peak: L{best_layer}\n{best_score:.3f}",
    xy=(best_layer, best_score),
    xytext=(best_layer + 1, best_score + 0.03),
    arrowprops=dict(arrowstyle="->")
)

ax.set_xlabel("Layer")
ax.set_ylabel("AUROC")
ax.set_title("Experiment 2: Layer-Profile Plot")
ax.set_ylim(0.45, 1.0)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp2_layer_profile.png", dpi=200)
plt.show()
print("Saved: exp2_layer_profile.png")

# --------------------------------------------------
# Figure 2: All 7 Metrics — Scalar AUROC Bar Chart
# --------------------------------------------------
metric_names = list(metric_scores.keys())
metric_aurocs = []

for name in metric_names:
    scores = metric_scores[name]
    auc = compute_scalar_auroc(scores, test_labels)
    metric_aurocs.append(auc if auc is not None else 0.0)

# color bars by performance
bar_colors = ["#2ecc71" if a >= 0.75 else "#f39c12" if a >= 0.6 else "#e74c3c"
              for a in metric_aurocs]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(metric_names, metric_aurocs, color=bar_colors, edgecolor="black", linewidth=0.7)

# value labels on bars
for bar, val in zip(bars, metric_aurocs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

ax.axhline(0.5, linestyle="--", linewidth=1, color="grey", label="Chance (0.5)")
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("AUROC")
ax.set_title("Experiment 2: All Metrics — AUROC Comparison")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
ax.legend()

# legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", label="Strong (≥0.75)"),
    Patch(facecolor="#f39c12", label="Moderate (0.60–0.75)"),
    Patch(facecolor="#e74c3c", label="Weak (<0.60)"),
]
ax.legend(handles=legend_elements + [plt.Line2D([0],[0], linestyle="--", color="grey", label="Chance")])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exp2_metric_comparison.png", dpi=200)
plt.show()
print("Saved: exp2_metric_comparison.png")

# --------------------------------------------------
# Print summary
# --------------------------------------------------
print("\n--- AUROC Summary ---")
for name, auc in sorted(zip(metric_names, metric_aurocs), key=lambda x: -x[1]):
    print(f"  {name:<30s} {auc:.4f}")