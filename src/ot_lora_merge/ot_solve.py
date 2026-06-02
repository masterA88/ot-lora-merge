"""Thin wrappers around POT (Python Optimal Transport).

Isolating POT here lets us swap exact / entropic / Gromov-Wasserstein and add log-domain
stabilization without touching the merge logic. Spec: experiment-spec.md §1.4, §1.6, §1.7.

The OT solves are all r x r (r <= 64) -> microseconds-milliseconds on CPU. No GPU.
"""
from __future__ import annotations

import numpy as np

try:
    import ot as _pot  # POT
    _HAS_POT = True
except ImportError:  # pragma: no cover - exercised only without POT installed
    _HAS_POT = False


def _require_pot() -> None:
    if not _HAS_POT:
        raise ImportError(
            "POT is required for OT solves. Install with: pip install POT==0.9.6"
        )


def sinkhorn(p, q, C, eps: float = 1e-2, num_iter: int = 1000,
             stop_thr: float = 1e-9) -> np.ndarray:
    """Entropic-regularized OT coupling P (Spec §1.4).

    The cost is normalized to max 1 so a single `eps` behaves consistently across layers/ranks.
    Plain Sinkhorn (fast) is used for moderate eps; the log-domain solver (stable but ~10x slower)
    only kicks in for small eps where plain Sinkhorn would underflow. For r x r problems this is
    milliseconds; if it still fails to converge we fall back to exact EMD.
    """
    _require_pot()
    p = np.ascontiguousarray(p, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    cmax = float(C.max())
    Cn = C / cmax if cmax > 0 else C
    method = "sinkhorn" if eps >= 5e-3 else "sinkhorn_log"
    with np.errstate(over="ignore", invalid="ignore"):
        P = _pot.sinkhorn(p, q, Cn, reg=eps, numItermax=num_iter,
                          stopThr=stop_thr, method=method, warn=False)
    if not np.all(np.isfinite(P)) or P.sum() <= 0:   # underflow guard -> exact
        return emd(p, q, C)
    return P


def emd(p, q, C) -> np.ndarray:
    """Exact OT coupling (eps -> 0 limit). Deterministic given distinct costs."""
    _require_pot()
    p = np.ascontiguousarray(p, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    return _pot.emd(p, q, C)


def coupling(p, q, C, eps: float | None = 1e-2, exact: bool = False) -> np.ndarray:
    """Dispatch: exact EMD if `exact` or eps is falsy, else Sinkhorn."""
    if exact or not eps:
        return emd(p, q, C)
    return sinkhorn(p, q, C, eps=eps)


def entropic_gw(D1, D2, p, q, eps: float = 5e-2, num_iter: int = 1000) -> np.ndarray:
    """Entropic Gromov-Wasserstein coupling for heterogeneous-rank merging (Spec §1.6).

    Matches two direction distributions by intra-distribution structure only (D1, D2),
    so adapters of *different* ranks can be coupled. Cost is r_i^2 * r_j^2 — still tiny.
    """
    _require_pot()
    D1 = np.ascontiguousarray(D1, dtype=np.float64)
    D2 = np.ascontiguousarray(D2, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    return _pot.gromov.entropic_gromov_wasserstein(
        D1, D2, p, q, loss_fun="square_loss", epsilon=eps, max_iter=num_iter
    )
