"""
Track B — Representation Metrics
==========================================
1. Attention Entropy     (Baseline 1)
2. Logit Confidence      (Baseline 2)
3. Cosine Drift
4. Mahalanobis Distance  (streaming fit via update/finalize)
5. Logit Lens Divergence
6. PCA Deviation         (streaming fit via update/finalize)
7. CIE top-3 layers      (layer-wise AUROC → pick top 3 → composite)

Each metric returns a per-token score for the response span.
Higher score = more likely hallucinated.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from extractor import HiddenStateBundle


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention Entropy  (Baseline 1)
# ─────────────────────────────────────────────────────────────────────────────

def attention_entropy(bundle: HiddenStateBundle, start: int, end: int) -> np.ndarray:
    """
    Mean entropy of attention distributions across all heads and layers,
    for each response token position.
    Returns shape: (end - start,)
    """
    attn = bundle.attentions          # (n_layers, n_heads, seq_len, seq_len)
    eps = 1e-10
    ent = -(attn * (attn + eps).log()).sum(dim=-1)  # (n_layers, n_heads, seq_len)
    ent_mean = ent.mean(dim=(0, 1))                  # (seq_len,)
    return ent_mean[start:end].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logit Confidence  (Baseline 2)
# ─────────────────────────────────────────────────────────────────────────────

def logit_confidence(bundle: HiddenStateBundle, start: int, end: int) -> np.ndarray:
    """
    1 - max(softmax(logits)) for each token.
    High value = low confidence = more likely hallucinated.
    """
    probs = F.softmax(bundle.logits, dim=-1)   # (seq_len, vocab)
    max_prob = probs.max(dim=-1).values        # (seq_len,)
    uncertainty = 1.0 - max_prob
    return uncertainty[start:end].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cosine Drift
# ─────────────────────────────────────────────────────────────────────────────

def cosine_drift(
    bundle: HiddenStateBundle,
    start: int,
    end: int,
    layer_range: Optional[Tuple[int, int]] = None,
    return_per_layer: bool = False,   # NEW
) -> np.ndarray:
    """
    For each response token t, cosine distance between hidden state at t
    and t-1.

    - Default: returns averaged drift over layer_range → shape [tokens]
    - If return_per_layer=True → returns per-layer drift → shape [layers, tokens]
    """

    hs = bundle.hidden_states  # (n_layers, seq_len, hidden_dim)
    n_layers = bundle.n_layers

    # Compute cosine drift for ALL layers at once
    per_layer_scores = []

    for t in range(start, end):
        if t == 0:
            per_layer_scores.append(np.zeros(n_layers))
            continue

        h_cur  = hs[:, t, :]       # (n_layers, hidden_dim)
        h_prev = hs[:, t - 1, :]

        cos_sim = F.cosine_similarity(h_cur, h_prev, dim=-1)  # (n_layers,)
        drift = (1.0 - cos_sim).cpu().numpy()                # (n_layers,)

        per_layer_scores.append(drift)

    # shape → (tokens, layers) → transpose
    per_layer_scores = np.stack(per_layer_scores, axis=0).T  # (layers, tokens)

    if return_per_layer:
        return per_layer_scores

    # --- ORIGINAL BEHAVIOR (preserved) ---
    if layer_range is None:
        lo = n_layers // 2
        hi = n_layers
    else:
        lo, hi = layer_range

    return per_layer_scores[lo:hi + 1].mean(axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Mahalanobis Distance  — STREAMING version
# ─────────────────────────────────────────────────────────────────────────────

class MahalanobisMetric:
    """
    Streaming fit: call update() on each training bundle, then finalize().
    Fit is done on faithful (label=0) samples only.
    """

    def __init__(self, layer_indices: Optional[list] = None):
        self.layer_indices = layer_indices
        # accumulators (filled during update)
        self._sum:    Optional[dict] = None   # layer → (D,)
        self._sum_sq: Optional[dict] = None   # layer → (D, D)
        self._count:  Optional[dict] = None   # layer → int
        # fitted parameters
        self.mu:      Optional[dict] = None
        self.inv_cov: Optional[dict] = None
        self._initialised = False

    def _init_layers(self, n_layers: int):
        if self.layer_indices is None:
            self.layer_indices = list(range(n_layers // 2, n_layers + 1))
        self._sum    = {l: None for l in self.layer_indices}
        self._sum_sq = {l: None for l in self.layer_indices}
        self._count  = {l: 0    for l in self.layer_indices}
        self._initialised = True

    def update(self, bundle: HiddenStateBundle, start: int, end: int, label: int):
        """Accumulate statistics from one training bundle (faithful only)."""
        if label != 0:
            return
        if not self._initialised:
            self._init_layers(bundle.n_layers)
        for l in self.layer_indices:
            hs = bundle.hidden_states[l, start:end, :].float()  # (T, D)
            if self._sum[l] is None:
                D = hs.shape[-1]
                self._sum[l]    = torch.zeros(D)
                self._sum_sq[l] = torch.zeros(D, D)
            self._sum[l]    += hs.sum(dim=0)
            self._sum_sq[l] += hs.T @ hs
            self._count[l]  += hs.shape[0]

    def finalize(self):
        """Compute μ and Σ⁻¹ from accumulators. Call after all update() calls."""
        assert self._initialised, "No update() calls before finalize()"
        self.mu      = {}
        self.inv_cov = {}
        fitted = 0
        for l in self.layer_indices:
            n = self._count[l]
            if n < 2 or self._sum[l] is None:
                continue
            mu  = self._sum[l] / n
            cov = self._sum_sq[l] / n - torch.outer(mu, mu)
            cov += torch.eye(cov.shape[0]) * 1e-2   # regularise
            self.mu[l]      = mu
            self.inv_cov[l] = torch.linalg.inv(cov)
            fitted += 1
        print(f"[Mahalanobis] Fitted {fitted} layers on {self._count.get(self.layer_indices[0], 0)} tokens (faithful only).")
        # free accumulators
        self._sum = self._sum_sq = self._count = None

    # legacy batch fit — kept for compatibility
    def fit(self, bundles: list, start_ends: list, labels: list):
        if not self._initialised:
            self._init_layers(bundles[0].n_layers)
        for bundle, (s, e), lbl in zip(bundles, start_ends, labels):
            self.update(bundle, s, e, lbl)
        self.finalize()

    def score(self, bundle: HiddenStateBundle, start: int, end: int) -> np.ndarray:
        assert self.mu is not None, "Call finalize() before score()"
        scores = []
        for t in range(start, end):
            dists = []
            for l in self.layer_indices:
                if l not in self.mu:
                    continue
                h     = bundle.hidden_states[l, t, :].float()
                delta = h - self.mu[l]
                dist  = (delta @ self.inv_cov[l] @ delta).clamp(min=0).sqrt().item()
                dists.append(dist)
            scores.append(np.mean(dists) if dists else 0.0)
        return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Logit Lens Divergence
# ─────────────────────────────────────────────────────────────────────────────

def logit_lens_divergence(
    bundle: HiddenStateBundle,
    start: int,
    end: int,
    model_lm_head: torch.nn.Linear,
    layer_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Project hidden states at intermediate layers through LM head.
    KL divergence vs final-layer distribution.
    NOTE: lm_head is moved to CPU temporarily to match bundle tensors.
    """
    n_layers = bundle.n_layers
    if layer_range is None:
        lo = 1                  #was: n_layers // 4 = layer 6, but start from layer 1, since factual retrieval starts there
        hi = n_layers - 1
    else:
        lo, hi = layer_range


    with torch.no_grad():
        final_hs = bundle.hidden_states[-1]                    # (seq_len, D)
        final_logits = model_lm_head(final_hs.float())           # (seq_len, vocab)
        final_log_probs = F.log_softmax(final_logits, dim=-1)  # (seq_len, vocab)

        scores = []
        for t in range(start, end):
            kl_vals = []
            for l in range(lo, hi):
                mid_hs     = bundle.hidden_states[l, t, :].unsqueeze(0).float()
                mid_logits = model_lm_head(mid_hs)
                mid_log_probs = F.log_softmax(mid_logits, dim=-1)
                kl = F.kl_div(
                    mid_log_probs,
                    final_log_probs[t].unsqueeze(0).exp(),
                    reduction="sum",
                    log_target=False,
                ).item()

                kl_vals.append(max(kl, 0.0))
            scores.append(float(np.mean(kl_vals)) if kl_vals else 0.0)

    return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA Deviation  — STREAMING version
# ─────────────────────────────────────────────────────────────────────────────

class PCADeviationMetric:
    def __init__(self, n_components: int = 50, layer_indices=None, reservoir_size: int = 5000): #CHANGED
        self.n_components   = n_components
        self.layer_indices  = layer_indices
        self.reservoir_size = reservoir_size
        self._reservoirs    = None   # layer → (reservoir_size, D) pre-allocated
        self._counts        = None   # layer → int (total seen, for reservoir sampling)
        self.components     = None
        self.mu             = None
        self._initialised   = False
        self._rng           = np.random.default_rng(42)

    def _init_layers(self, n_layers: int): #CHANGED
        if self.layer_indices is None:
            self.layer_indices = list(range(n_layers // 2, n_layers + 1))
        self._reservoirs = {}
        self._counts     = {l: 0 for l in self.layer_indices}
        self._initialised = True

    def update(self, bundle: HiddenStateBundle, start: int, end: int, label: int): #CHANGED
        if label != 0:
            return
        if not self._initialised:
            self._init_layers(bundle.n_layers)
        for l in self.layer_indices:
            hs = bundle.hidden_states[l, start:end, :].float()  # (T, D)
            for vec in hs:                                        # one token at a time
                n = self._counts[l]
                if n < self.reservoir_size:
                    if l not in self._reservoirs:
                        D = vec.shape[0]
                        self._reservoirs[l] = torch.zeros(self.reservoir_size, D)
                    self._reservoirs[l][n] = vec.detach()
                else:
                    # reservoir sampling: replace random earlier entry
                    j = int(self._rng.integers(0, n + 1))
                    if j < self.reservoir_size:
                        self._reservoirs[l][j] = vec.detach()
                self._counts[l] += 1

    def finalize(self): #CHANGED
        assert self._initialised
        self.components = {}
        self.mu         = {}
        for l in self.layer_indices:
            if l not in self._reservoirs or self._counts[l] == 0:
                continue
            n_filled = min(self._counts[l], self.reservoir_size)
            X  = self._reservoirs[l][:n_filled].float()
            mu = X.mean(dim=0)
            X_c = X - mu
            _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
            k = min(self.n_components, Vt.shape[0])
            self.components[l] = Vt[:k]
            self.mu[l]         = mu
        print(f"[PCA] Fitted {len(self.components)} layers.")
        self._reservoirs = None   # free memory

    # legacy batch fit — kept for compatibility
    def fit(self, bundles: list, start_ends: list, labels: list):
        if not self._initialised:
            self._init_layers(bundles[0].n_layers)
        for bundle, (s, e), lbl in zip(bundles, start_ends, labels):
            self.update(bundle, s, e, lbl)
        self.finalize()

    def score(self, bundle: HiddenStateBundle, start: int, end: int) -> np.ndarray:
        assert self.components is not None, "Call finalize() before score()"
        scores = []
        for t in range(start, end):
            errs = []
            for l in self.layer_indices:
                if l not in self.components:
                    continue
                h    = bundle.hidden_states[l, t, :].float()
                h_c  = h - self.mu[l]
                V    = self.components[l]          # (k, D)
                proj = V @ h_c                     # (k,)
                recon = V.T @ proj                 # (D,)
                residual = (h_c - recon).norm().item()
                errs.append(residual)
            scores.append(np.mean(errs) if errs else 0.0)
        return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CIE top-3 layers  (Causal Intervention Effect proxy)
# ─────────────────────────────────────────────────────────────────────────────

class CIETopLayersMetric:
    """
    Proxy CIE metric:
      1. During fit, record per-layer mean hidden-state norm for faithful
         and hallucinated samples separately.
      2. Rank layers by |μ_halluc − μ_faithful| and pick top-3.
      3. Score = cosine drift averaged over those top-3 layers only.

    This approximates the Causal Intervention Effect (layer importance)
    without requiring full activation patching.
    """

    def __init__(self):
        self._faith_norms:  Optional[dict] = None   # layer → list[float]
        self._halluc_norms: Optional[dict] = None
        self.top_layers:    Optional[List[int]] = None
        self._initialised = False
        self._n_layers = None

    def _init(self, n_layers: int):
        self._n_layers = n_layers
        layers = list(range(n_layers + 1))
        self._faith_norms  = {l: [] for l in layers}
        self._halluc_norms = {l: [] for l in layers}
        self._initialised = True

    def update(self, bundle: HiddenStateBundle, start: int, end: int, label: int):
        if not self._initialised:
            self._init(bundle.n_layers)
        target = self._halluc_norms if label == 1 else self._faith_norms
        for l in range(bundle.n_layers + 1):
            hs = bundle.hidden_states[l, start:end, :].float()
            drift = F.cosine_similarity(hs[1:], hs[:-1], dim=-1)
            target[l].append(drift.var().item())

    def finalize(self):
        assert self._initialised
        diffs = {}
        for l in self._faith_norms:
            f = self._faith_norms[l]
            h = self._halluc_norms[l]
            if f and h:
                diffs[l] = abs(np.mean(h) - np.mean(f)) / (np.std(h + f) + 1e-6)
        # pick top 3 layers by |μ_halluc − μ_faithful|
        ranked = sorted(diffs, key=lambda x: diffs[x], reverse=True)
        self.top_layers = ranked[:3]
        print(f"[CIE] Top-3 discriminative layers: {self.top_layers}  "
              f"Δnorm: {[round(diffs[l],4) for l in self.top_layers]}")
        self._faith_norms = self._halluc_norms = None
    
    def score(self, bundle, start, end):
        """For hallucination detection — top-3 layers only (correct behaviour)."""
        assert self.top_layers is not None, "Call finalize() first"
        per_layer = cosine_drift(bundle, start, end, return_per_layer=True)
        selected = np.stack([per_layer[l] for l in self.top_layers], axis=0)
        return selected.mean(axis=0)   # (T,)

    def score_all_layers(self, bundle, start, end):
        """For E3 layer localisation — returns per-layer scores, shape (n_layers, T)."""
        assert self.top_layers is not None
        return cosine_drift(bundle, start, end, return_per_layer=True)  # (n_layers, T)
