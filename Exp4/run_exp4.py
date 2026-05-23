import json
import numpy as np
import torch
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/medihse/Documents/NLP_Project/FINAL/Exp1_2")

WINDOW = [-3, -2, -1, 0, 1]

# Load E1/E2 results
data = torch.load("../outputs/exp1_2_results.pt", map_location="cpu", weights_only=False)
pts = data['per_token_scores']
sample_ids = [int(x) for x in data['test_sample_ids']]

# Load processed (for onset tokens t)
with open("e4_processed.json") as f:
    processed = json.load(f)

METRICS = {
    "cosine":     "Cosine Drift",
    "mahalanobis":"Mahalanobis",
    "logit_lens": "Logit Lens Div.",
    "pca_dev":    "PCA Deviation",
    "cie":        "CIE top-3 layers"
}

all_results = {off: {m: [] for m in METRICS} for off in WINDOW}

for list_idx, sample_idx in enumerate(sample_ids):
    if sample_idx >= len(processed):
        continue
    sample = processed[sample_idx]
    t_list = sample.get("t", [])
    seq_len = sample.get("seq_len", None)

    for t in t_list:
        if t is None:
            continue
        t = int(t)

        for off in WINDOW:
            pos = t + off
            if pos < 0:
                continue

            for metric_key, metric_name in METRICS.items():
                token_scores = pts[metric_name][list_idx]  # (seq_len,)
                if pos >= len(token_scores):
                    continue
                val = float(token_scores[pos])
                if not np.isnan(val):
                    all_results[off][metric_key].append(val)

# Print sample counts
print("Sample counts per offset:")
for metric_key in METRICS:
    counts = [len(all_results[off][metric_key]) for off in WINDOW]
    print(f"  {metric_key}: {dict(zip(WINDOW, counts))}")

# Aggregate means
means = {off: {m: np.mean(v) if v else np.nan
               for m, v in all_results[off].items()}
         for off in WINDOW}

# Aggregate standard deviations
sds = {off: {m: np.std(v) if len(v) > 1 else 0.0
             for m, v in all_results[off].items()}
       for off in WINDOW}

# Print table
pos_labels = {-3:"t-3", -2:"t-2", -1:"t-1", 0:"t", 1:"t+1"}
print(f"\n{'Offset':>6} {'Cosine':>10} {'Mahal':>10} {'LogitLens':>12} {'PCA_dev':>10} {'CIE':>10}")
for off in WINDOW:
    m = means[off]
    print(f"{pos_labels[off]:>6} {m['cosine']:>10.4f} {m['mahalanobis']:>10.4f} "
          f"{m['logit_lens']:>12.4f} {m['pca_dev']:>10.4f} {m['cie']:>10.4f}")

# Mann-Whitney U: pool t-3+t-2 vs t+1
print("\n=== Mann-Whitney U (t-3+t-2 vs t+1) ===")
for metric_key in METRICS:
    a = all_results[-3][metric_key] + all_results[-2][metric_key]
    b = all_results[1][metric_key]
    if len(a) > 1 and len(b) > 1:
        stat, p = mannwhitneyu(a, b, alternative="greater")
        sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
        print(f"  {metric_key:>12}: U={stat:.1f}, p={p:.4f} {sig}")

# Plot
plt.figure(figsize=(9, 5))
#for metric_key in METRICS:
#    vals = np.array([means[off][metric_key] for off in WINDOW])
#    vals = (vals - np.nanmean(vals)) / np.nanstd(vals)  # z-score
#    plt.plot(WINDOW, vals, marker='o', label=metric_key)
#    peak_idx = np.nanargmax(vals)
#    peak_x = WINDOW[peak_idx]
#    peak_y = vals[peak_idx]
#    plt.scatter(peak_x, peak_y, s=80, zorder=5, edgecolors='black', linewidths=0.8)

for metric_key in METRICS:
    vals    = np.array([means[off][metric_key] for off in WINDOW])
    sd_vals = np.array([sds[off][metric_key]   for off in WINDOW])

    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    vals_z    = (vals - vmin)    / (vmax - vmin + 1e-8)   # min-max normalization
    sd_vals_z =  sd_vals         / (vmax - vmin + 1e-8)   # min-max normalization

    line, = plt.plot(WINDOW, vals_z, marker='o', label=metric_key)
    color = line.get_color()
    plt.fill_between(WINDOW,
                     vals_z - sd_vals_z,
                     vals_z + sd_vals_z,
                     alpha=0.15, color=color)

    peak_idx = np.nanargmax(vals_z)
    plt.scatter(WINDOW[peak_idx], vals_z[peak_idx],
                s=80, zorder=5, edgecolors='black', linewidths=0.8, color=color)

plt.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
plt.axvline(x=0, color='red', linestyle='--', label='onset t')
plt.xticks(WINDOW, [pos_labels[o] for o in WINDOW])
plt.xlabel("Position relative to onset (t)")
plt.ylabel("Drift / Divergence")
plt.title("E4: Temporal Precedence of Representation Drift")
plt.legend(loc='upper right')
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("../outputs/e4_temporal_plot.png", dpi=150)
print("\nPlot saved.")