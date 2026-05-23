import torch

# load your results
data = torch.load("exp1_2_results.pt", map_location="cpu", weights_only=False)

# ---------------------------
# Extract baseline + best
# ---------------------------

baseline = data["individual"]["Attention Entropy (B1)"]["AUROC"]

best_auroc = -1
best_name = None

def get_auroc(val):
    if isinstance(val, dict) and "AUROC" in val:
        return val["AUROC"]
    return None

# check both individual + composite
for section in ["individual", "composite"]:
    for name, val in data[section].items():
        auroc = get_auroc(val)
        if auroc is not None and auroc > best_auroc:
            best_auroc = auroc
            best_name = f"{section} -> {name}"

# ---------------------------
# SOTA values (given)
# ---------------------------
redeep = 0.82
lumina = 0.87

# ---------------------------
# Gap calculation
# ---------------------------
def gap_closed(your_score, baseline, sota):
    return (your_score - baseline) / (sota - baseline)

gap_redeep = gap_closed(best_auroc, baseline, redeep)
gap_lumina = gap_closed(best_auroc, baseline, lumina)

# ---------------------------
# Print results
# ---------------------------
print("\n===== E8 RESULTS =====\n")

print(f"Baseline (Attention Entropy): {baseline:.4f}")
print(f"Best Method: {best_name}")
print(f"Best AUROC: {best_auroc:.4f}\n")

print("SOTA Comparison:")
print(f"ReDeEP (~0.82): Gap Closed = {gap_redeep:.4f} ({gap_redeep*100:.2f}%)")
print(f"LUMINA (~0.87): Gap Closed = {gap_lumina:.4f} ({gap_lumina*100:.2f}%)")

print("\n======================\n")