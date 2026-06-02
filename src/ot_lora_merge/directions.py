"""ΔW -> rank-r singular directions, and rank-R refactor back to LoRA form.

This is the ONLY place where a LoRA update ΔW = (alpha/r) B A is turned into the
empirical direction distribution (U, sigma, V, p) that every downstream module consumes.
All work is r x r — we never materialize the full m x n matrix.

Spec: experiment-spec.md §1.1–1.2, §1.7 step 1, and the rank-R refactor in step 4.
"""
from __future__ import annotations

import numpy as np


def _canonical_sign(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kill SVD sign-flip nondeterminism: force the largest-|component| of each U
    column positive, flipping the paired V column to match. (Determinism note, §1.7.)"""
    for k in range(U.shape[1]):
        idx = int(np.argmax(np.abs(U[:, k])))
        if U[idx, k] < 0:
            U[:, k] = -U[:, k]
            V[:, k] = -V[:, k]
    return U, V


def extract_directions(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Singular directions of a LoRA update.

    Args:
        A: (r, n) LoRA down-projection.
        B: (m, r) LoRA up-projection.
        alpha: LoRA scaling; if given, scale = alpha / r. Ignored if `scale` is set.
        scale: explicit scale on ΔW = scale * B @ A. Defaults to 1.0.

    Returns:
        U:     (m, r) left singular vectors  (Stiefel)
        sigma: (r,)   singular values (descending)
        V:     (n, r) right singular vectors (Stiefel)
        p:     (r,)   mass per direction, p_k = sigma_k / sum_j sigma_j

    Guarantee: U @ diag(sigma) @ V.T  ==  scale * B @ A  (up to fp error).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, r = B.shape
    r2, n = A.shape
    if r != r2:
        raise ValueError(f"rank mismatch: B has r={r}, A has r={r2}")
    if scale is None:
        scale = (alpha / r) if alpha is not None else 1.0

    # Thin QR of each factor: B = Q_B R_B, A^T = Q_A R_A  (so A = R_A^T Q_A^T).
    Q_B, R_B = np.linalg.qr(B)        # Q_B (m,r), R_B (r,r)
    Q_A, R_A = np.linalg.qr(A.T)      # Q_A (n,r), R_A (r,r)

    # ΔW = scale * Q_B (R_B R_A^T) Q_A^T  =  Q_B M Q_A^T, all r x r below.
    M = scale * (R_B @ R_A.T)         # (r, r)
    Ut, s, Vt = np.linalg.svd(M)      # M = Ut diag(s) Vt

    U = Q_B @ Ut                      # (m, r)
    V = Q_A @ Vt.T                    # (n, r)
    U, V = _canonical_sign(U, V)

    sigma = np.asarray(s, dtype=np.float64)
    total = sigma.sum()
    p = sigma / total if total > 0 else np.full(r, 1.0 / r)
    return U, sigma, V, p


def reconstruct(U: np.ndarray, sigma: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Dense ΔW = U diag(sigma) V^T. Used internally / in tests; (m x n) — avoid at scale."""
    return (U * sigma) @ V.T


def refactor(dW: np.ndarray, R: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor a dense merged update back to rank-R LoRA form (A*, B*).

    Returns:
        A_star: (R, n)
        B_star: (m, R)
    with B_star @ A_star == best rank-R approximation of dW.
    (Spec §1.7 step 4: split sqrt(Sigma) symmetrically across the two factors.)
    """
    U, s, Vt = np.linalg.svd(np.asarray(dW, dtype=np.float64), full_matrices=False)
    R = min(R, s.shape[0])
    U_R = U[:, :R]
    s_R = s[:R]
    V_R = Vt[:R, :]
    sqrt_s = np.sqrt(s_R)
    B_star = U_R * sqrt_s              # (m, R)
    A_star = sqrt_s[:, None] * V_R     # (R, n)
    return A_star, B_star
