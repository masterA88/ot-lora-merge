"""M-align: transport-align each task's directions to an anchor, then combine. (Spec §1.5.)

The barycentric projection maps task-t's singular directions onto the anchor's atoms via the
OT plan P^{at}; the transported singular mass at anchor atom k is p_a[k] * sum(sigma_t), so that
a task identical to the anchor maps to itself exactly (the identity guarantee used in tests).
"""
from __future__ import annotations

import numpy as np


def pick_anchor(dirs) -> int:
    """Anchor = task with the largest total singular mass (Spec §1.5)."""
    return int(np.argmax([d[1].sum() for d in dirs]))


def barycentric_map(U_t, V_t, P_at, p_a):
    """Map task-t directions into the anchor frame.

    Ũ[:,k] = sum_l (P_at[k,l] / p_a[k]) U_t[:,l]   (and likewise Ṽ).
    Returns mapped (U_mapped: m x r_a, V_mapped: n x r_a).
    """
    W = P_at / p_a[:, None]      # (r_a, r_t) barycentric weights
    U_mapped = U_t @ W.T          # (m, r_a)
    V_mapped = V_t @ W.T          # (n, r_a)
    return U_mapped, V_mapped


def aligned_factors(U_t, sigma_t, V_t, p_t, U_a, p_a, P_at):
    """Factored form of task t transported into the anchor frame: (U_m, V_m, sigma_tilde).

    Transported singular mass at anchor atom k:  sigma~_k = p_a[k] * sum(sigma_t).
    Dense update = (U_m * sigma_tilde) @ V_m.T. Returning factors lets the barycenter loop avoid
    ever forming / SVD-ing the dense m x n matrix (keeps everything thin).
    """
    U_m, V_m = barycentric_map(U_t, V_t, P_at, p_a)   # (m, r_a), (n, r_a)
    sigma_tilde = p_a * float(sigma_t.sum())           # (r_a,)
    return U_m, V_m, sigma_tilde


def aligned_update(U_t, sigma_t, V_t, p_t, U_a, p_a, P_at):
    """Dense ΔW of task t, transported into the anchor frame.

    For an identical task (P = diag(p_a)), this returns exactly task t's own ΔW.
    """
    U_m, V_m, sigma_tilde = aligned_factors(U_t, sigma_t, V_t, p_t, U_a, p_a, P_at)
    return (U_m * sigma_tilde) @ V_m.T


def thin_svd_from_factors(L, R, target_rank):
    """Top-`target_rank` SVD of (L @ R.T) without forming the m x n product.

    L: (m, K), R: (n, K). Uses QR on each factor then a small K x K SVD. Returns (U, sigma, V).
    """
    Q_L, R_L = np.linalg.qr(L)        # (m, K), (K, K)
    Q_R, R_R = np.linalg.qr(R)        # (n, K), (K, K)
    Uu, s, Vt = np.linalg.svd(R_L @ R_R.T)
    k = min(target_rank, s.shape[0])
    U = Q_L @ Uu[:, :k]
    V = Q_R @ Vt[:k, :].T
    return U, s[:k], V
