"""Ground-cost matrices between rank-r direction distributions, and GW structure matrices.

Spec: experiment-spec.md §1.3 (ground costs) and §1.6 (Gromov-Wasserstein structure).
All inputs are Stiefel factors (U: m x r, V: n x r); all outputs are tiny r_i x r_j.
Sign ambiguity is handled by absolute values |<.,.>|.
"""
from __future__ import annotations

import numpy as np


def paired_cost(U_i, V_i, U_j, V_j) -> np.ndarray:
    """Default: C[k,l] = 1 - |<u_ik,u_jl>| * |<v_ik,v_jl>|  (paired left&right collinearity)."""
    GU = np.abs(U_i.T @ U_j)   # (r_i, r_j)
    GV = np.abs(V_i.T @ V_j)
    return 1.0 - GU * GV


def left_cost(U_i, V_i, U_j, V_j) -> np.ndarray:
    """C[k,l] = 1 - |<u_ik,u_jl>|  (left/U-only; the 'merge-B' regime)."""
    return 1.0 - np.abs(U_i.T @ U_j)


def right_cost(U_i, V_i, U_j, V_j) -> np.ndarray:
    """C[k,l] = 1 - |<v_ik,v_jl>|  (right/V-only; the 'merge-A' regime)."""
    return 1.0 - np.abs(V_i.T @ V_j)


def grassmann_cost(U_i, V_i, U_j, V_j) -> np.ndarray:
    """Chordal/Grassmann distance on the 1-dim subspaces spanned by paired directions.
    sin^2 angle = 1 - cos^2, on the product of left & right collinearity (sign-invariant)."""
    GU = np.abs(U_i.T @ U_j)
    GV = np.abs(V_i.T @ V_j)
    c2 = (GU * GV) ** 2
    return np.sqrt(np.clip(1.0 - c2, 0.0, 1.0))


def gw_structure(U, V) -> np.ndarray:
    """Intra-task structure matrix D^{(t)}[k,k'] = 1 - |<u_k,u_k'>||<v_k,v_k'>| (Spec §1.6).
    Used by the Gromov-Wasserstein path for heterogeneous-rank merging."""
    GU = np.abs(U.T @ U)
    GV = np.abs(V.T @ V)
    return 1.0 - GU * GV


COSTS = {
    "paired": paired_cost,
    "left": left_cost,
    "right": right_cost,
    "grassmann": grassmann_cost,
}
