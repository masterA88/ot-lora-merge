"""OT-TIES: interference-resolved merge in an OT-chosen shared subspace. (M2 headline method.)

Motivation (validated empirically, results/m2*.md): merging LoRA adapters by *averaging* their
transported directions (align / barycenter) plateaus at the simple-average floor (~0.62 avg-norm-acc),
because averaging blurs rather than *resolves* sign interference. The methods that win (KnOTS,
Core-Space) instead (i) build a shared subspace and (ii) apply TIES-style sign-election + magnitude
preservation inside it. KnOTS chooses that subspace by an SVD of the stacked task updates.

This module keeps the TIES combine but chooses the shared subspace by **optimal transport** — the
top-k left singular basis of the Wasserstein barycenter of the task direction-distributions. On the
8-task ViT-B/32 KnOTS-LoRA benchmark this OT basis beats the SVD basis at every scaling tested
(ot 0.708 vs svd 0.687 at the peak; +0.7..+1.9% across lambda), instantiating the thesis
"OT replaces KnOTS's ad-hoc SVD alignment".

Algorithm (per layer, all CPU / numpy):
  1. dW_t = scale_t * B_t @ A_t                         for each task t
  2. shared basis U = top-k left singular vectors of the OT barycenter of {dW_t}    (m x k)
  3. project each task into the basis:  S_t = U^T dW_t                              (k x n)
  4. TIES on {S_t}: trim each task to its top-`keep` fraction by |.|, sign-elect by the sum's sign,
     disjoint-mean over agreeing tasks, scale by `lam`                              (k x n)
  5. dW* = lam_already_applied? -> U @ TIES                                          (m x n, full update)

Public API:
  ot_ties_layer(adapters, ...)  -> dense dW* (m x n)   for one layer
  ot_ties_model(layer_adapters) -> {layer: dW*}        across a model

Defaults (k=128, keep=0.2, lam=0.7, cost='paired', eps=1e-2) are the empirical peak from M2.
The merged update is full-rank-up-to-k (not a rank-r LoRA); apply it directly to the base weight.
"""
from __future__ import annotations

import numpy as np

from .barycenter import wasserstein_barycenter
from .directions import extract_directions


def ties_combine(coeffs, keep: float = 0.2, lam: float = 0.7) -> np.ndarray:
    """TIES merge of stacked per-task coefficient blocks.

    Args:
        coeffs: list of T arrays, each shape (k, n) — the per-task projections.
        keep:   fraction of largest-|.| entries to keep per task (trim the rest to 0).
        lam:    scaling on the merged result.
    Returns:
        merged (k, n).
    Steps (Yadav 2023): trim -> sign-elect (sign of the sum) -> disjoint-mean over agreeing tasks.
    """
    S = np.stack(coeffs, 0)                       # (T, k, n)
    if keep < 1.0:
        T = S.shape[0]
        a = np.abs(S).reshape(T, -1)
        th = np.quantile(a, 1.0 - keep, axis=1, keepdims=True)
        S = S * (a >= th).reshape(S.shape)
    gamma = np.sign(S.sum(0))                      # (k, n) elected sign
    agree = (np.sign(S) == gamma) & (S != 0)       # (T, k, n)
    num = (S * agree).sum(0)                        # (k, n)
    den = agree.sum(0).astype(float)
    den[den == 0] = 1.0
    return lam * (num / den)


def _ot_basis(adapters, k: int, cost: str, eps: float) -> np.ndarray:
    """Top-k left singular basis of the OT (Wasserstein) barycenter of the task updates."""
    dirs = [
        extract_directions(ad["A"], ad["B"], alpha=ad.get("alpha"), scale=ad.get("scale"))
        for ad in adapters
    ]
    w = np.full(len(dirs), 1.0 / len(dirs))
    dW_bar = wasserstein_barycenter(dirs, w, cost=cost, eps=eps, target_rank=k)
    U, _, _ = np.linalg.svd(dW_bar, full_matrices=False)
    return U[:, :k]


def ot_ties_layer(
    adapters,
    k: int = 128,
    keep: float = 0.2,
    lam: float = 0.7,
    cost: str = "paired",
    eps: float | None = 1e-2,
) -> np.ndarray:
    """Merge one layer's adapters via OT-TIES. Returns dense dW* (m x n).

    adapters: list of {"A": (r,n), "B": (m,r), "alpha"|"scale": float}.
    The merged update is the full-rank-up-to-k matrix to add to the base layer weight.
    """
    dWs = [
        (ad.get("scale", (ad["alpha"] / ad["A"].shape[0]) if ad.get("alpha") is not None else 1.0))
        * (np.asarray(ad["B"], float) @ np.asarray(ad["A"], float))
        for ad in adapters
    ]
    if len(adapters) == 1:                          # identity short-circuit
        return dWs[0]
    U = _ot_basis(adapters, k, cost, eps)           # (m, k)
    coeffs = [U.T @ dW for dW in dWs]               # T x (k, n)
    merged = ties_combine(coeffs, keep=keep, lam=lam)
    return U @ merged                               # (m, n)


def ot_ties_model(layer_adapters: dict, **kwargs) -> dict:
    """Apply ot_ties_layer across {layer_name: [adapter, ...]}. Returns {layer_name: dW*}."""
    return {name: ot_ties_layer(ads, **kwargs) for name, ads in layer_adapters.items()}
