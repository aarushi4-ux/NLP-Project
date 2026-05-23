"""
Track B — Experiments 1 & 2  (Fixed + Freeze for E3–E8)
=======================================================================
Fixes applied:
  1. Memory management: del hidden states + torch.cuda.empty_cache() each step
  2. Device fix: lm_head stays on CPU during logit_lens_divergence
  3. MahalanobisMetric / PCADeviationMetric now use streaming update/finalize
  4. CIETopLayersMetric added (required for E1/E2 table)
  5. lm_head cached as CPU copy once — not re-copied every token
  6. BUG FIX: `steps` list moved before `metric_aurocs` dict that depends on it
  7. FREEZE: saves all state needed for E3–E8 into exp1_2_results.pt so
     teammates never need to rerun the expensive extraction pipeline

What is frozen and why:
  ┌───────────────────────────┬──────────────────────────────────────────────────┐
  │ Key in .pt file           │ Used by                                          │
  ├───────────────────────────┼──────────────────────────────────────────────────┤
  │ metric_scores             │ E3 temporal precedence (raw per-sample scores)   │
  │ test_labels               │ All downstream experiments                       │
  │ layer_profile             │ E2 layer plot, E6 FFN vs attention localisation  │
  │ layer_scores_raw          │ E3 per-layer per-sample values for temporal plot │
  │ per_token_scores          │ E3 — token-level scores needed for t-3..t+1      │
  │ mahalanobis.mu/inv_cov    │ E5 zero-shot HaluEval (no re-fitting allowed)    │
  │ pca.components/mean       │ E5 zero-shot HaluEval                            │
  │ cie_full object           │ E5 zero-shot HaluEval                            │
  │ normalization.min/max     │ E5 — same scaler must be reused on HaluEval      │
  │ weights                   │ E5 — same weights must be reused on HaluEval     │
  │ individual_results        │ E4/E5/E6/E8 AUROC tables                        │
  │ composite_results         │ E4/E5/E8 AUROC tables                            │
  │ cie_top_layers            │ E3 / E6 layer localisation                       │
  │ train_sample_ids          │ E5 leakage check                                 │
  │ test_sample_ids           │ E5 leakage check                                 │
  └───────────────────────────┴──────────────────────────────────────────────────┘
"""

import argparse
import gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import time
import warnings
import random
from tqdm import tqdm
from collections import defaultdict
warnings.filterwarnings("ignore")

from data_loader import load_ragtruth_official_splits
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


# ─────────────────────────────────────────────────────────────────────────────
# Experiment order — defined ONCE at module level so nothing uses it before
# it exists (this was the bug in the previous version)
# ─────────────────────────────────────────────────────────────────────────────

STEPS = [
    "Attention Entropy (B1)",
    "Logit Confidence (B2)",
    "Cosine Drift",
    "Mahalanobis",
    "Logit Lens Div.",
    "PCA Deviation",
    "CIE top-3 layers",
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(scores, labels, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if labels[idx].sum() in (0, n):
            continue
        boot.append(roc_auc_score(labels[idx], scores[idx]))
    auc = roc_auc_score(labels, scores)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"AUROC": round(auc, 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)}


def compute_f1_span(scores, labels):
    best = 0
    for thr in np.percentile(scores, np.arange(10, 91, 5)):
        f1 = f1_score(labels, (scores >= thr).astype(int), zero_division=0)
        best = max(best, f1)
    return round(best, 4)


def expected_calibration_error(scores, labels, bins=10):
    smin, smax = scores.min(), scores.max()
    probs = (scores - smin) / (smax - smin + 1e-8)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return round(float(ece), 4)


# ─────────────────────────────────────────────────────────────────────────────
# CPU copy of lm_head — built once, reused by all logit_lens calls
# ─────────────────────────────────────────────────────────────────────────────

def make_cpu_lm_head(model_lm_head: torch.nn.Linear) -> torch.nn.Linear:
    """Detach and move lm_head weights to CPU. Done once per run."""
    cpu_head = torch.nn.Linear(
        model_lm_head.in_features,
        model_lm_head.out_features,
        bias=model_lm_head.bias is not None,
    )
    cpu_head.weight = torch.nn.Parameter(model_lm_head.weight.detach().cpu())
    if model_lm_head.bias is not None:
        cpu_head.bias = torch.nn.Parameter(model_lm_head.bias.detach().cpu())
    cpu_head.eval()
    return cpu_head


# ─────────────────────────────────────────────────────────────────────────────
# Streaming pass — fit OR score
# Returns an extra `per_token_scores` dict for E3 temporal precedence
# ─────────────────────────────────────────────────────────────────────────────

def stream_scores(
    samples,
    extractor,
    maha=None,
    pca=None,
    cie=None,
    model_lm_head=None,
    fit_mode=False,
):
    """
    Streaming pass over `samples`.

    Returns (fit_mode=True):
        None, None, None, None

    Returns (fit_mode=False):
        scores            : dict[metric_name -> np.ndarray shape (N,)]
        labels            : np.ndarray shape (N,)
        layer_scores      : dict[layer_idx(1-based) -> list of per-sample mean drifts]
        per_token_scores  : dict[metric_name -> list of np.ndarray (T,)] — one array
                           per sample, length = number of response tokens.
                           Used by E3 to align t-3..t+1 relative to first hallucinated token.
        attn_layer_scores : np.ndarray
                            Per-layer attention-based anomaly scores (averaged over tokens), used for
                            analyzing which attention layers contribute to hallucination signals.
        ffn_layer_scores : np.ndarray
                            Per-layer feedforward (FFN) deviation scores (averaged over tokens), used
                            to identify layers associated with knowledge storage or hallucination behavior.
    """

    # Layer-wise drift accumulators (1-indexed to match n_layers)
    n_layers = extractor.n_layers
    layer_scores      = {l: [] for l in range(1, n_layers + 1)}
    attn_layer_scores = {l: [] for l in range(1, n_layers + 1)}   # ADD
    ffn_layer_scores  = {l: [] for l in range(1, n_layers + 1)}   # ADD

    # Per-token storage — for E3 temporal precedence
    # Each entry is a 1-D numpy array of length = response token count
    per_token_scores = {k: [] for k in STEPS}

    def agg(x):
        return float(x.mean()) if len(x) > 0 else 0.0

    scores = {k: [] for k in STEPS}
    labels = []

    t0 = time.time()
    total = len(samples)

    # add these accumulators before the sample loop
    hidden_dim = extractor.model.config.hidden_size
    layer_hs_sum_hal = {l: np.zeros(hidden_dim, dtype=np.float64) for l in range(1, n_layers + 1)}
    layer_hs_sum_fai = {l: np.zeros(hidden_dim, dtype=np.float64) for l in range(1, n_layers + 1)}
    layer_hs_count   = {"hal": {l: 0 for l in range(1, n_layers + 1)},
                        "fai": {l: 0 for l in range(1, n_layers + 1)}}

    _warned_components = False
    cie_all_layer_scores = [] 

    for i, s in enumerate(samples):

        prompt = extractor.build_prompt(s)
        bundle = extractor.extract(prompt)

        assert bundle.hidden_states is not None, "Hidden states are missing!"

        start, end = extractor.response_token_range(bundle.tokens, s.response)
        if end <= start:
            end = min(start + 10, len(bundle.tokens))

        if fit_mode:
            maha.update(bundle, start, end, s.label)
            pca.update(bundle, start, end, s.label)
            cie.update(bundle, start, end, s.label)

        else:
            # ── cosine drift per-layer (shape: n_layers × T) ─────────────
            per_layer_drifts = cosine_drift(
                bundle, start, end, return_per_layer=True
            )

            # mid-to-late layers only for the composite score
            lo_layer = int(bundle.n_layers * 0.65)
            hi_layer = bundle.n_layers - 1
            global_vals = per_layer_drifts[lo_layer: hi_layer + 1].mean(axis=0)

            # ── per-token vectors for each metric (shape: T,) ────────────
            ae_tok  = attention_entropy(bundle, start, end)          # (T,)
            lc_tok  = logit_confidence(bundle, start, end)           # (T,)
            mh_tok  = maha.score(bundle, start, end)                 # (T,)
            ll_tok  = logit_lens_divergence(
                bundle, start, end, model_lm_head)                   # (T,)
            pca_tok = pca.score(bundle, start, end)                  # (T,)
            cie_tok = cie.score(bundle, start, end)                  # (T,) — used as metric
            cie_all_layers   = cie.score_all_layers(bundle, start, end) # (n_layers, T) — for E3
            cie_all_layer_scores.append(cie_all_layers)              # list of (n_layers, T) arrays

            # ── aggregate to sample-level scores ─────────────────────────
            scores["Attention Entropy (B1)"].append(agg(ae_tok))
            scores["Logit Confidence (B2)"].append(agg(lc_tok))
            scores["Cosine Drift"].append(agg(global_vals))
            scores["Mahalanobis"].append(agg(mh_tok))
            scores["Logit Lens Div."].append(agg(ll_tok))
            scores["PCA Deviation"].append(agg(pca_tok))
            scores["CIE top-3 layers"].append(agg(cie_tok))
            labels.append(s.label)

            # ── store raw token arrays for E3 ────────────────────────────
            def _save(arr):
                a = np.asarray(arr)
                return a.flatten() if a.ndim > 1 else a

            per_token_scores["Attention Entropy (B1)"].append(_save(ae_tok))
            per_token_scores["Logit Confidence (B2)"].append(_save(lc_tok))
            per_token_scores["Cosine Drift"].append(_save(global_vals))
            per_token_scores["Mahalanobis"].append(_save(mh_tok))
            per_token_scores["Logit Lens Div."].append(_save(ll_tok))
            per_token_scores["PCA Deviation"].append(_save(pca_tok))
            per_token_scores["CIE top-3 layers"].append(_save(cie_tok))

            # ── per-layer drift for E6 layer profile ─────────────────────
            for l in range(bundle.n_layers):
                ld = per_layer_drifts[l]
                layer_scores[l + 1].append(
                    float(ld.mean()) if len(ld) > 0 else 0.0
                )
            
            # E3 prep — attn vs FFN drift per layer per sample

            # Inside the loop, replace has_components with:
            has_components = (
                hasattr(bundle, "attn_outputs")
                and hasattr(bundle, "ffn_outputs")
                and bundle.attn_outputs is not None
                and bundle.ffn_outputs is not None
            )
            if not has_components and not _warned_components:
                warnings.warn(
                    "bundle.attn_outputs not found — attn/FFN layer scores will be zeros. "
                    "E6 decomposition will not be meaningful.",
                    RuntimeWarning,
                )
                _warned_components = True


            for l in range(bundle.n_layers):
                if has_components:
                    sl = slice(start, end)
                    attn_l = bundle.attn_outputs[l, sl, :]
                    ffn_l  = bundle.ffn_outputs[l, sl, :]
                    hs_l   = bundle.hidden_states[l, sl, :]

                    attn_cos = F.cosine_similarity(attn_l, hs_l, dim=-1).mean().item()
                    ffn_cos  = F.cosine_similarity(ffn_l,  hs_l, dim=-1).mean().item()
                else:
                    attn_cos = 0.0
                    ffn_cos  = 0.0

                attn_layer_scores[l + 1].append(attn_cos)
                ffn_layer_scores[l + 1].append(ffn_cos)
            

            # ── mean hidden states per layer for E3 patching ─────────────────
            for l in range(bundle.n_layers):
                hs_mean = bundle.hidden_states[l, start:end, :].mean(axis=0)  # (D,)
                hs_np   = hs_mean.detach().cpu().float().numpy()

                if s.label == 1:  # hallucinated
                    layer_hs_sum_hal[l + 1] += hs_np
                    layer_hs_count["hal"][l + 1] += 1
                else:             # faithful
                    layer_hs_sum_fai[l + 1] += hs_np
                    layer_hs_count["fai"][l + 1] += 1
                
        # ── Memory cleanup ────────────────────────────────────────────────
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        done = i + 1
        eta = (elapsed / done) * (total - done) / 60
        print(
            f"{done}/{total}  |  "
            f"elapsed {elapsed/60:.1f}min  |  "
            f"remaining {eta:.1f}min",
            end="\r",
        )
    
    # Wrap the layer_mean_states construction:
    if not fit_mode:
        layer_mean_states = {}
        for l in range(1, n_layers + 1):
            hal_vec = (
                layer_hs_sum_hal[l] / layer_hs_count["hal"][l]
                if layer_hs_count["hal"][l] > 0
                else np.zeros(hidden_dim)
            )
            fai_vec = (
                layer_hs_sum_fai[l] / layer_hs_count["fai"][l]
                if layer_hs_count["fai"][l] > 0
                else np.zeros(hidden_dim)
            )
            layer_mean_states[l] = {
                "hallucinated": hal_vec.astype(np.float32),
                "faithful":     fai_vec.astype(np.float32),
                "delta":        (hal_vec - fai_vec).astype(np.float32),
            }
    else:
        layer_mean_states = None


    print()

    if fit_mode:
        return None, None, None, None, None, None, None

    for k in scores:
        scores[k] = np.array(scores[k])

    return (
    scores,
    np.array(labels),
    layer_scores,
    per_token_scores,
    attn_layer_scores,
    ffn_layer_scores,
    layer_mean_states,
    cie_all_layer_scores
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(args):

    print("Loading RAGTruth...")
    train_s, test_s = load_ragtruth_official_splits(args.response, args.source)
    print(f"Train: {len(train_s)} | Test: {len(test_s)}")

    extractor = HiddenStateExtractor("gpt2-medium")

    # Build CPU lm_head once — used by logit_lens_divergence
    model_lm_head = make_cpu_lm_head(extractor.model.lm_head)

    # ─────────────────────────────────────────
    # Fit Mahalanobis, PCA, CIE (streaming)
    # ─────────────────────────────────────────

    print("\nFitting metrics (streaming over train set)...")

    maha = MahalanobisMetric()
    pca  = PCADeviationMetric(n_components=50)
    cie  = CIETopLayersMetric()

    stream_scores(train_s, extractor, maha, pca, cie, model_lm_head, fit_mode=True)

    maha.finalize()
    pca.finalize()
    cie.finalize()

    assert maha.mu is not None,        "Mahalanobis finalize() failed"
    assert pca.components is not None, "PCA finalize() failed"
    assert cie.top_layers is not None, "CIE finalize() failed"

    print("\nScoring train set for normalization...")

    
    # SUBSET - stratified by label
    def stratified_sample(samples, n):
        by_label = defaultdict(list)
        for s in samples:
            by_label[s.label].append(s)
        result = []
        for label_samples in by_label.values():
            k = max(1, int(n * len(label_samples) / len(samples)))
            result.extend(random.sample(label_samples, min(k, len(label_samples))))
        return result

    subset_size = min(300, len(train_s))   # increase from 300 to 500
    subset = stratified_sample(train_s, subset_size)

    train_metric_scores, train_labels, _, _, _, _, _, _ = stream_scores(
        subset, extractor, maha, pca, cie, model_lm_head, fit_mode=False
    )

    train_stack = np.column_stack([train_metric_scores[s] for s in STEPS])
    norm_min = train_stack.min(axis=0)
    norm_max = train_stack.max(axis=0)

    # ─────────────────────────────────────────
    # Test scoring
    # ─────────────────────────────────────────

    print("\nScoring test set...")

    n_layers = extractor.n_layers
    hidden_dim = extractor.model.config.hidden_size

    metric_scores, test_labels, layer_scores_raw, per_token_scores, attn_layer_scores, ffn_layer_scores, layer_mean_states, cie_all_layer_scores = stream_scores(
        test_s, extractor, maha, pca, cie, model_lm_head, fit_mode=False, 
    )


    # ── Per-layer AUROC profile (used for E2 layer plot and E6) ──────────────
    layer_profile = {}
    for l, sc in layer_scores_raw.items():
        sc_arr = np.array(sc)
        if sc_arr.std() > 0:
            layer_profile[l] = float(roc_auc_score(test_labels, sc_arr))
        else:
            layer_profile[l] = 0.5   # degenerate — flat scores

    # ─────────────────────────────────────────
    # Experiment 1 — Individual metrics
    # ─────────────────────────────────────────

    print("\n" + "=" * 75)
    print("EXPERIMENT 1 — Individual Metric Results")
    print("=" * 75)
    print(f"{'Metric':<28} {'AUROC':>6}  {'CI_lo':>6}  {'CI_hi':>6}  {'F1':>6}  {'Spearman':>9}  {'ECE':>6}")
    print("-" * 75)

    individual_results = {}

    for name, sc in metric_scores.items():
        res = bootstrap_auroc(sc, test_labels)
        f1  = compute_f1_span(sc, test_labels)
        rho = spearmanr(sc, test_labels).statistic
        ece = expected_calibration_error(sc, test_labels)

        individual_results[name] = {
            "AUROC":    res["AUROC"],
            "CI_lo":    res["CI_lo"],
            "CI_hi":    res["CI_hi"],
            "F1":       f1,
            "Spearman": round(float(rho), 4),
            "ECE":      ece,
        }

        print(
            f"{name:<28} "
            f"{res['AUROC']:>6.4f}  "
            f"[{res['CI_lo']:.4f},{res['CI_hi']:.4f}]  "
            f"{f1:>6.4f}  "
            f"{float(rho):>9.4f}  "
            f"{ece:>6.4f}"
        )

    # ─────────────────────────────────────────
    # Experiment 2 — Composite (incremental)
    # ─────────────────────────────────────────

    # actually AUROC-proportional (computed after individual_results is populated)
    raw_weights = np.array([
        individual_results[step]["AUROC"] for step in STEPS
    ])
    # Subtract 0.5 so random-chance metrics get near-zero weight
    raw_weights = np.clip(raw_weights - 0.5, 0.0, None)
    weights = raw_weights / (raw_weights.sum() + 1e-8)

    print("\n" + "=" * 75)
    print("EXPERIMENT 2 — Composite (incremental)")
    print("=" * 75)
    print(f"{'After adding':<28} {'AUROC':>6}  {'CI_lo':>6}  {'CI_hi':>6}  {'F1':>6}  {'Spearman':>9}  {'ECE':>6}")
    print("-" * 75)

    accumulated       = []
    composite_results = {}

    for i, step in enumerate(STEPS):
        accumulated.append(metric_scores[step])
        stack  = np.column_stack(accumulated)
        normed = (stack - norm_min[:i + 1]) / (norm_max[:i + 1] - norm_min[:i + 1] + 1e-8)
        normed = np.clip(normed, 0.0, 1.0)

        w    = weights[:i + 1]
        w    = w / w.sum()
        comp = (normed * w).sum(axis=1)

        res = bootstrap_auroc(comp, test_labels)
        f1  = compute_f1_span(comp, test_labels)
        rho = spearmanr(comp, test_labels).statistic
        ece = expected_calibration_error(comp, test_labels)

        composite_results[step] = {
            "AUROC":    res["AUROC"],
            "CI_lo":    res["CI_lo"],
            "CI_hi":    res["CI_hi"],
            "F1":       f1,
            "Spearman": round(float(rho), 4),
            "ECE":      ece,
        }

        print(
            f"+ {step:<26} "
            f"{res['AUROC']:>6.4f}  "
            f"[{res['CI_lo']:.4f},{res['CI_hi']:.4f}]  "
            f"{f1:>6.4f}  "
            f"{float(rho):>9.4f}  "
            f"{ece:>6.4f}"
        )

    # ── Leakage check ─────────────────────────────────────────────────────────
    print("\n── Leakage Check ──────────────────────────────────────")
    print("  Mahalanobis μ/Σ  : fitted on TRAIN only")
    print("  PCA components   : fitted on TRAIN only")
    print("  CIE layer ranks  : fitted on TRAIN only")
    print(f"  Train samples used for fit : {len(train_s)}")
    print(f"  Test samples scored        : {len(test_s)}")
    overlap = len(
        set(s.sample_id for s in train_s) & set(s.sample_id for s in test_s)
    )
    print(f"  Label overlap (should be 0): {overlap}")
    print("────────────────────────────────────────────────────────")

    # ─────────────────────────────────────────
    # Save — everything teammates need for E3–E8
    # ─────────────────────────────────────────


    save_dict = {
        # ── E1/E2 tables ─────────────────────────────────────────────────────
        "individual":       individual_results,
        "composite":        composite_results,

        # ── Raw scores & labels (needed by E3, E4, E5, E6, E8) ───────────────
        # metric_scores: dict[name -> np.ndarray (N,)]  — one value per test sample
        "metric_scores":    metric_scores,
        "test_labels": test_labels.astype(np.int64),

        # ── Per-token scores for E3 temporal precedence ──────────────────────
        # per_token_scores: dict[name -> list of np.ndarray (T_i,)]
        # To align at t, look up s.first_hallucinated_token_idx for each sample
        "per_token_scores": per_token_scores,

        # ── Layer-level information for E2 plot and E6 FFN/attn decomp ───────
        # layer_profile: dict[layer_idx -> AUROC float]
        "layer_profile":    layer_profile,
        # layer_scores_raw: dict[layer_idx -> list of per-sample mean drifts]
        "layer_scores_raw": {k: np.array(v)
                             for k, v in layer_scores_raw.items()},

        # ── CIE localisation ─────────────────────────────────────────────────
        "cie_top_layers":   cie.top_layers,          # list[int] — 1-based layer indices
        "cie_top3": {
            "layers": cie.top_layers,
            "AUROC":  individual_results["CIE top-3 layers"]["AUROC"],
        },
        "cie_all_layer_scores": cie_all_layer_scores,  # list of (n_layers, T_i) — for E3

        # ── Frozen fitted parameters for E5 zero-shot HaluEval ───────────────
        # CRITICAL: must NOT be re-fitted on HaluEval — pass these directly
        # to MahalanobisMetric.load() / PCADeviationMetric.load() etc.
        "mahalanobis": {
            "mu":      maha.mu,          # dict[layer_idx -> np.ndarray (D,)]
            "inv_cov": maha.inv_cov,     # dict[layer_idx -> np.ndarray (D,D)]
        },
        "pca": {
            "components": pca.components,  # np.ndarray (n_components, D) or dict per layer
            "mean":       pca.mu,          # np.ndarray (D,) or dict per layer
        },
        # The full CIE object (top_layers attribute is what matters for scoring)
        "cie_full":     cie,

        # ── Normalisation constants — reuse on HaluEval (E5) ─────────────────
        # Apply: normed = (raw - norm_min) / (norm_max - norm_min + 1e-8)
        "normalization": {
            "min":     norm_min,    # np.ndarray (n_metrics,)
            "max":     norm_max,    # np.ndarray (n_metrics,)
            "steps":   STEPS,       # same ordering as min/max columns
        },

        # ── Composite weights — reuse on HaluEval (E5) ───────────────────────
        "weights":      weights,    # np.ndarray (n_metrics,) — AUROC-proportional

        # ── Sample IDs for leakage audit (E5) ────────────────────────────────
        "train_sample_ids": [s.sample_id for s in train_s],
        "test_sample_ids":  [s.sample_id for s in test_s],

        # ── E3 activation patching prerequisites ─────────────────────────────
        # attn/ffn cosine alignment per layer per sample
        # shape after conversion: dict[layer_idx -> np.ndarray (N,)]
        "attn_layer_scores": {k: np.array(v)
                              for k, v in attn_layer_scores.items()},
        "ffn_layer_scores":  {k: np.array(v)
                              for k, v in ffn_layer_scores.items()},

        # per-layer AUROC split by component — E3 CIE-by-component table
        "attn_layer_profile": {
            l: float(roc_auc_score(test_labels, np.array(v)))
            if np.array(v).std() > 0 else 0.5
            for l, v in attn_layer_scores.items()
        },
        "ffn_layer_profile": {
            l: float(roc_auc_score(test_labels, np.array(v)))
            if np.array(v).std() > 0 else 0.5
            for l, v in ffn_layer_scores.items()
        },

        # faithful/hallucinated mean hidden states per layer — needed for
        # patching source/target pool construction in E3
        # shape: dict[layer_idx -> {"faithful": np.ndarray (D,), "hallucinated": np.ndarray (D,)}]
        "layer_mean_states": layer_mean_states,
    }

    torch.save(save_dict, "exp1_2_results.pt")
    print("\nSaved → exp1_2_results.pt")
    print("\nFrozen keys available for teammates:")
    for k in save_dict:
        print(f"  {k}")

    # ─────────────────────────────────────────
    # Mark estimate
    # ─────────────────────────────────────────

    final_auroc = list(composite_results.values())[-1]["AUROC"]
    b1_auroc    = individual_results["Attention Entropy (B1)"]["AUROC"]
    b2_auroc    = individual_results["Logit Confidence (B2)"]["AUROC"]
    beats_both  = final_auroc > b1_auroc and final_auroc > b2_auroc

    print("\n── Mark estimate (Exp 1 & 2) ──────────────────────────")
    print(f"  Final composite AUROC : {final_auroc:.4f}")
    print(f"  Beats both baselines  : {beats_both} (B1={b1_auroc:.4f}, B2={b2_auroc:.4f})")
    if   final_auroc >= 0.70 and beats_both: mark = 4
    elif final_auroc >= 0.63 and beats_both: mark = 3
    elif final_auroc >= 0.55:               mark = 2
    elif final_auroc >= 0.48:               mark = 1
    else:                                   mark = 0
    print(f"  Estimated mark        : {mark}/4")
    print("────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", required=True)
    parser.add_argument("--source",   required=True)
    args = parser.parse_args()
    run_experiments(args)
