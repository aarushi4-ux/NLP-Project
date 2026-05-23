import argparse
import pandas as pd
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_dataset
from pair_selector import build_pairs
from patch_runner import run_patching_experiment, save_activations
import sys
sys.path.insert(0, "/Users/medihse/Documents/NLP_Project/FINAL/Exp1_2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="../outputs/metrics_per_example.csv")
    parser.add_argument("--pt",         default="../outputs/exp1_2_results.pt")
    parser.add_argument("--out",        default="../outputs/exp3_results.csv")
    parser.add_argument("--max_pairs",  type=int, default=147)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--cie_mode",   default="last", choices=["last", "mean", "max"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"[Setup] Using device: {device}")

    print("[Setup] Loading GPT-2 Medium...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    model.eval()

    dataset = load_dataset(args.csv, args.pt, tokenizer, max_length=args.max_length, device=device)
    pairs = build_pairs(dataset, max_pairs=args.max_pairs)

    save_activations(pairs, model, device, out_dir="activations")

    print(f"[Exp3] Running patching on {len(pairs)} pairs...")
    results = run_patching_experiment(model, pairs, device)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"[Exp3] Saved {len(df)} rows to {args.out}")

    def plot_cie(df, out_path="../outputs/patching_plot.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        directions = ["f_to_h", "h_to_f"]
        titles     = ["Faithful → Hallucinated", "Hallucinated → Faithful"]
        components = ["attn", "ffn"]
        colors     = {"attn": "#4C72B0", "ffn": "#DD8452"}

        for ax, direction, title in zip(axes, directions, titles):
            sub = df[df["direction"] == direction]
            layers = sorted(sub["layer"].unique())

            for comp in components:
                means, lo, hi = [], [], []
                for l in layers:
                    vals = sub[(sub["layer"] == l) & (sub["component"] == comp)]["cie"].values
                    if len(vals) == 0:
                        means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                        continue
                    m = np.mean(vals)
                    se = 1.96 * np.std(vals) / np.sqrt(len(vals))
                    means.append(m)
                    lo.append(se)
                    hi.append(se)

                ax.errorbar(
                    layers, means,
                    yerr=[lo, hi],
                    label=comp.upper(),
                    color=colors[comp],
                    linewidth=2,
                    capsize=3,
                    marker="o",
                    markersize=4,
                )

            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("CIE Score", fontsize=11)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle("CIE by Component and Patching Direction", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[Plot] Saved → {out_path}")

    plot_cie(df)

    # Quick summary by layer band
    n_layers = model.config.n_layer  # 24 for gpt2-medium
    def band(layer):
        pct = layer / n_layers
        if pct < 0.25:  return "early (0-25%)"
        elif pct < 0.75: return "mid (25-75%)"
        else:            return "late (75-100%)"

    df["band"] = df["layer"].apply(band)
    summary = (df.groupby(["direction", "band", "component"])["cie"]
                 .agg(["mean", "std", "count"])
                 .reset_index())

    df["significant"] = df["p_value"] < 0.001
    summary = (df.groupby(["direction", "band", "component"])["cie"]
                .agg(["mean", "std", "count"])
                .reset_index())
    summary["significant"] = (df.groupby(["direction", "band", "component"])["significant"]
                                .mean()
                                .values)
    print("\n=== CIE Summary ===")
    print(summary.to_string(index=False))

    print("\n=== p-values ===")
    df["significant"] = df["p_value"] < 0.001
    print(df.groupby(["component","band"])["significant"].mean().reset_index(name="significant"))


    # --- Layer Drift Analysis for E4 ---
    print("\n=== Top Drift Layers (E3 → E4 handoff) ===")

    # Mean CIE per layer and component, averaged across directions and pairs
    layer_summary = (
        df.groupby(["layer", "component"])["cie"]
        .mean()
        .reset_index()
        .sort_values("cie", ascending=False)
    )

    print(layer_summary.to_string(index=False))

    # Separate top layers per component
    top_attn = (
        layer_summary[layer_summary["component"] == "attn"]
        .nlargest(5, "cie")["layer"]
        .tolist()
    )
    top_ffn = (
        layer_summary[layer_summary["component"] == "ffn"]
        .nlargest(5, "cie")["layer"]
        .tolist()
    )

    print(f"\nTop 5 attention layers: {top_attn}")
    print(f"Top 5 FFN layers:       {top_ffn}")
    print(f"Combined (union):       {sorted(set(top_attn + top_ffn))}")

    # Save for E4 to load directly
    e4_config = {
        "top_attn_layers": top_attn,
        "top_ffn_layers": top_ffn,
        "top_layers_combined": sorted(set(top_attn + top_ffn))
    }

    with open("../outputs/e4_layer_config.json", "w") as f:
        json.dump(e4_config, f, indent=2)

    print("\n[E3] Saved e4_layer_config.json for downstream use")


if __name__ == "__main__":
    main()