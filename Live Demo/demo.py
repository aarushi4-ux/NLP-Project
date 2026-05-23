"""
demo.py
====================
Live demo: takes a single passage + context, extracts hidden states,
computes all 5 representation metrics, outputs token-level scores.

Usage:
    python demo.py

    # Or pass evaluator's input directly
    python demo.py \
    --context "Your passage here" \
    --question "Question here" \
    --response "Model response here" \
    --pt path/to/exp1_2_results.pt
"""

import argparse
import numpy as np
import torch
import sys
sys.path.insert(0, "/Users/medihse/Documents/NLP_Project/FINAL/Exp1_2")

from composite import make_cpu_lm_head
from extractor import HiddenStateExtractor
from metrics import (
    attention_entropy,
    logit_confidence,
    cosine_drift,
    MahalanobisMetric,
    logit_lens_divergence,
    PCADeviationMetric,
    CIETopLayersMetric,
)

# ── Default demo input (swap out for evaluator's input) ──────────────────────
# Test 1 — Fully correct
DEFAULT_CONTEXT = "The Eiffel Tower is located in Paris, France. It was constructed in 1889 as the entrance arch for the 1889 World's Fair. It stands 330 metres tall."

DEFAULT_QUESTION = "Where is the Eiffel Tower and when was it built?"

DEFAULT_RESPONSE = "The Eiffel Tower was built in 1889 and is located in Paris, France. It stands 330 metres tall."



def load_frozen_metrics(pt_path):
    """
    Load μₗ and Σₗ (and PCA/CIE) from exp1_2_results.pt.
    CRITICAL: these were fitted on train split only — no re-fitting here.
    """
    print(f"[Demo] Loading frozen parameters from {pt_path} ...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # ── Mahalanobis: restore μₗ and Σₗ ──────────────────────────────────────
    # μₗ  = data["mahalanobis"]["mu"]      — dict[layer -> (D,) tensor]
    # Σₗ  = data["mahalanobis"]["inv_cov"] — dict[layer -> (D,D) tensor]
    maha = MahalanobisMetric()
    maha.mu      = data["mahalanobis"]["mu"]
    maha.inv_cov = data["mahalanobis"]["inv_cov"]
    maha.layer_indices = list(maha.mu.keys())

    # ── PCA: restore components and mean ─────────────────────────────────────
    pca = PCADeviationMetric()
    pca.components   = data["pca"]["components"]
    pca.mu           = data["pca"]["mean"]
    pca.layer_indices = list(pca.components.keys())

    # ── CIE: restore top layers ───────────────────────────────────────────────
    cie = data["cie_full"]   # full object with top_layers already set

    # ── Normalisation constants ───────────────────────────────────────────────
    norm_min = data["normalization"]["min"]
    norm_max = data["normalization"]["max"]
    weights  = data["weights"]

    return maha, pca, cie, norm_min, norm_max, weights


def run_demo(context, question, response, pt_path="exp1_2_results.pt"):

    # ── Load model ────────────────────────────────────────────────────────────
    extractor = HiddenStateExtractor("gpt2-medium")
    model_lm_head = make_cpu_lm_head(extractor.model.lm_head)

    # ── Load frozen μₗ / Σₗ / PCA / CIE ─────────────────────────────────────
    maha, pca, cie, norm_min, norm_max, weights = load_frozen_metrics(pt_path)

    # ── Build prompt ──────────────────────────────────────────────────────────
    class _Sample:
        def __init__(self, context, question, response):
            self.context  = context
            self.question = question
            self.response = response

    sample = _Sample(context, question, response)
    prompt = extractor.build_prompt(sample)
    print("\n" + "=" * 60)
    print("INPUT SUMMARY")
    print("=" * 60)
    print(f"CONTEXT: {context.strip()}\n")
    print(f"QUESTION: {question.strip()}\n")
    print(f"RESPONSE: {response.strip()}\n")
    print("=" * 60 + "\n")

    # ── Extract hidden states ─────────────────────────────────────────────────
    print("[Demo] Extracting hidden states...")
    bundle = extractor.extract(prompt)
    answer_tokens = extractor.tokenizer.tokenize("Answer:")
    
    for i in range(len(bundle.tokens) - len(answer_tokens)):
        if bundle.tokens[i:i+len(answer_tokens)] == answer_tokens:
            start = i + len(answer_tokens)
            break
    else:
        start = len(bundle.tokens) // 2

    end = len(bundle.tokens)
    print(f"[Demo] Response token range: [{start}, {end}]  ({end - start} tokens)\n")

    response_tokens = bundle.tokens[start:end]

    # ── Compute all 5 metrics (token-level) ───────────────────────────────────
    print("[Demo] Computing metrics...\n")

    ae_tok  = attention_entropy(bundle, start, end)
    lc_tok  = logit_confidence(bundle, start, end)
    cd_tok  = cosine_drift(bundle, start, end)
    mh_tok  = maha.score(bundle, start, end)
    ll_tok  = logit_lens_divergence(bundle, start, end, model_lm_head)
    pca_tok = pca.score(bundle, start, end)
    cie_tok = cie.score(bundle, start, end)

    # ── Print token-level table ───────────────────────────────────────────────
    METRICS = {
        "Attn Entropy":    ae_tok,
        "Logit Conf":      lc_tok,
        "Cosine Drift":    cd_tok,
        "Mahalanobis":     mh_tok,
        "Logit Lens Div":  ll_tok,
        "PCA Deviation":   pca_tok,
        "CIE top-3":       cie_tok,
    }

    #-----Display metrics for each token---------------------------
    header = f"{'Token':<20}" + "".join(f"{k:>16}" for k in METRICS)
    print(header)
    print("-" * len(header))
    for i, tok in enumerate(response_tokens):
        row = f"{tok:<20}"
        for arr in METRICS.values():
            val = arr[i] if i < len(arr) else 0.0
            row += f"{val:>16.4f}"
        print(row)

    #-----Display hallucination info for each token-----------------
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL HALLUCINATION FLAGS")
    print("=" * 60)

    flagged_tokens = []
    for i, tok in enumerate(response_tokens):
        # compute raw scores for this single token
        raw = np.array([
            ae_tok[i], lc_tok[i], cd_tok[i],
            mh_tok[i], ll_tok[i], pca_tok[i], cie_tok[i]
        ])
        # normalise using frozen training constants
        normed = np.clip((raw - norm_min) / (norm_max - norm_min + 1e-8), 0, 1)
        # weighted composite using learned weights
        w = weights / weights.sum()
        composite = float((normed * w).sum())

        if composite > 0.6:
            level = "HIGH"
        elif composite > 0.4:
            level = "UNCERTAIN"
        elif composite > 0.25:
            level = "WEAK"
        else:
            level = ""

        if level:
            flagged_tokens.append((tok, level, composite))
            print(f"  {tok:<20} [{level:<9}]  composite: {composite:.4f}")

    if not flagged_tokens:
        print("  No tokens flagged.")

    # ── Summary ───────────────────────────────────────────────────────────────
    high_count = sum(1 for _, lvl, _ in flagged_tokens if lvl == "HIGH")
    unc_count  = sum(1 for _, lvl, _ in flagged_tokens if lvl == "UNCERTAIN")
    weak_count = sum(1 for _, lvl, _ in flagged_tokens if lvl == "WEAK")

    print(f"\n  Flagged: {len(flagged_tokens)} tokens  "
          f"({high_count} HIGH, {unc_count} UNCERTAIN, {weak_count} WEAK)")

    if high_count >= 1:
        overall = "HIGH"
    elif unc_count >= 2:
        overall = "UNCERTAIN"
    elif unc_count == 1 and weak_count >= 2:
        overall = "UNCERTAIN"
    elif len(flagged_tokens) >= 3:
        overall = "UNCERTAIN"
    else:
        overall = "LOW"

    print(f"  Overall hallucination likelihood: {overall}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context",  default=DEFAULT_CONTEXT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--response", default=DEFAULT_RESPONSE)
    parser.add_argument("--pt",       default="../outputs/exp1_2_results.pt",
                        help="../outputs/exp1_2_results.pt")
    args = parser.parse_args()

    run_demo(args.context, args.question, args.response, args.pt)