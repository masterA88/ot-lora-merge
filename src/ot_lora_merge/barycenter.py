"""M-bary: free-support Wasserstein barycenter of the T direction distributions. (Spec §1.5 headline.)

Initialize nu from the simple-average update's directions, then alternate:
  (i)  Sinkhorn transport P^t from nu to each mu_t,
  (ii) map each task into the current nu frame (barycentric projection, transported mass) and
       accumulate the lambda-weighted dense update,
  (iii) re-extract nu's paired directions from that dense update by SVD.
Re-extracting from the dense accumulator keeps the left/right singular pairing consistent (an
independent QR of U and V would decouple them). Returns a dense ΔW_merged.
"""
from __future__ import annotations

import numpy as np

from .align import aligned_factors, thin_svd_from_factors
from .cost import COSTS
from .ot_solve import coupling


def wasserstein_barycenter(
    dirs,
    weights,
    cost: str = "paired",
    eps: float | None = 1e-2,
    exact: bool = False,
    target_rank: int | None = None,
    max_iter: int = 12,
    tol: float = 1e-6,
):
    """Free-support Wasserstein barycenter over rank-r direction distributions.

    Args:
        dirs:    list of (U, sigma, V, p) per task.
        weights: (T,) barycenter weights lambda_t (sum to 1).
        cost:    ground-cost key in cost.COSTS.
        eps:     Sinkhorn regularization (None/0 or exact=True -> exact EMD).
        target_rank: number of barycenter atoms R (default = max task rank).
    Returns:
        dense ΔW_merged (m x n).
    """
    cost_fn = COSTS[cost]
    T = len(dirs)
    weights = np.asarray(weights, dtype=np.float64)
    m = dirs[0][0].shape[0]
    n = dirs[0][2].shape[0]
    R = target_rank or max(d[1].shape[0] for d in dirs)

    # --- init nu from the simple-average update's top-R directions (factored, no dense SVD) ---
    L = np.hstack([weights[t] * (dirs[t][0] * dirs[t][1]) for t in range(T)])  # (m, sum r_t)
    Rm = np.hstack([dirs[t][2] for t in range(T)])                            # (n, sum r_t)
    U_bar, sigma_bar, V_bar = thin_svd_from_factors(L, Rm, R)
    p_bar = _masses(sigma_bar)
    prev_sigma_sum = sigma_bar.sum()

    for _ in range(max_iter):
        L_blocks, R_blocks = [], []
        for t in range(T):
            U_t, s_t, V_t, p_t = dirs[t]
            C = cost_fn(U_bar, V_bar, U_t, V_t)              # (R, r_t)
            P = coupling(p_bar, p_t, C, eps=eps, exact=exact)
            U_m, V_m, sig = aligned_factors(U_t, s_t, V_t, p_t, U_bar, p_bar, P)
            L_blocks.append(weights[t] * (U_m * sig))        # (m, R)
            R_blocks.append(V_m)                              # (n, R)
        L = np.hstack(L_blocks)
        Rm = np.hstack(R_blocks)
        U_bar, sigma_bar, V_bar = thin_svd_from_factors(L, Rm, R)
        p_bar = _masses(sigma_bar)
        s_sum = sigma_bar.sum()
        if abs(s_sum - prev_sigma_sum) <= tol * (prev_sigma_sum + 1e-12):
            break
        prev_sigma_sum = s_sum

    return (U_bar * sigma_bar) @ V_bar.T


def _masses(sigma):
    tot = sigma.sum()
    return sigma / tot if tot > 0 else np.full(sigma.shape[0], 1.0 / sigma.shape[0])
