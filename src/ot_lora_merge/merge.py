"""Top-level OT-LoRA-Merge orchestrator (Algorithm, Spec §1.7).

merge_layer: merge a list of adapters for ONE layer into rank-R LoRA factors (A*, B*).
merge_model: apply merge_layer across a dict of {layer_name: [adapters]}.

An adapter is a dict: {"A": (r,n), "B": (m,r), "alpha": float | "scale": float}.
Everything here is numpy + CPU; no torch dependency, so it is unit-testable offline.
"""
from __future__ import annotations

import numpy as np

from .align import aligned_update, pick_anchor
from .barycenter import wasserstein_barycenter
from .cost import COSTS, gw_structure
from .directions import extract_directions, reconstruct, refactor
from .ot_solve import coupling, entropic_gw


def _extract(adapters):
    return [
        extract_directions(ad["A"], ad["B"], alpha=ad.get("alpha"), scale=ad.get("scale"))
        for ad in adapters
    ]


def merge_layer(
    adapters,
    mode: str = "align",
    cost: str = "paired",
    eps: float | None = 1e-2,
    exact: bool = False,
    weights=None,
    target_rank: int | None = None,
    gw_eps: float = 5e-2,
):
    """Merge one layer's adapters.

    Args:
        adapters: list of {"A","B", "alpha"|"scale"}.
        mode:     "align" (M-align), "barycenter" (M-bary), or "gw" (heterogeneous-rank).
        cost:     ground-cost key ("paired"|"left"|"right"|"grassmann").
        eps:      Sinkhorn regularization; None/0 or exact=True -> exact EMD.
        weights:  (T,) combine weights; default uniform.
        target_rank: output rank R; default = max input rank.
    Returns:
        (A_star (R,n), B_star (m,R)).
    """
    dirs = _extract(adapters)
    T = len(dirs)
    ranks = [d[1].shape[0] for d in dirs]
    R = target_rank or max(ranks)

    if T == 1:                                   # identity short-circuit
        U, s, V, _ = dirs[0]
        return refactor(reconstruct(U, s, V), R)

    if weights is None:
        weights = np.full(T, 1.0 / T)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

    homogeneous = len(set(ranks)) == 1
    if mode == "gw" or (not homogeneous and mode == "align"):
        dW = _gw_merge(dirs, weights, R, gw_eps)
    elif mode == "barycenter":
        dW = wasserstein_barycenter(
            dirs, weights, cost=cost, eps=eps, exact=exact, target_rank=R
        )
    elif mode == "align":
        dW = _align_merge(dirs, weights, cost, eps, exact)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return refactor(dW, R)


def _align_merge(dirs, weights, cost, eps, exact):
    cost_fn = COSTS[cost]
    a = pick_anchor(dirs)
    U_a, s_a, V_a, p_a = dirs[a]
    m, n = U_a.shape[0], V_a.shape[0]
    dW = np.zeros((m, n))
    for t, (U_t, s_t, V_t, p_t) in enumerate(dirs):
        if t == a:
            dW += weights[t] * reconstruct(U_a, s_a, V_a)
            continue
        C = cost_fn(U_a, V_a, U_t, V_t)
        P = coupling(p_a, p_t, C, eps=eps, exact=exact)
        dW += weights[t] * aligned_update(U_t, s_t, V_t, p_t, U_a, p_a, P)
    return dW


def _gw_merge(dirs, weights, R, gw_eps):
    """Heterogeneous-rank merge via Gromov-Wasserstein (Spec §1.6) — the structural moat.

    Anchor = highest-rank task (closest to target R). Each other task is GW-coupled to the
    anchor by intra-task structure only, then barycentric-mapped into the anchor frame.
    """
    a = int(np.argmax([d[1].shape[0] for d in dirs]))
    U_a, s_a, V_a, p_a = dirs[a]
    D_a = gw_structure(U_a, V_a)
    m, n = U_a.shape[0], V_a.shape[0]
    dW = np.zeros((m, n))
    for t, (U_t, s_t, V_t, p_t) in enumerate(dirs):
        if t == a:
            dW += weights[t] * reconstruct(U_a, s_a, V_a)
            continue
        D_t = gw_structure(U_t, V_t)
        P = entropic_gw(D_a, D_t, p_a, p_t, eps=gw_eps)   # (r_a, r_t)
        dW += weights[t] * aligned_update(U_t, s_t, V_t, p_t, U_a, p_a, P)
    return dW


def merge_model(layer_adapters: dict, **kwargs) -> dict:
    """Merge every layer. `layer_adapters`: {layer_name: [adapter, ...]}.
    Returns {layer_name: (A_star, B_star)}."""
    return {name: merge_layer(ads, **kwargs) for name, ads in layer_adapters.items()}
