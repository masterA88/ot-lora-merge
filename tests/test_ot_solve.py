"""Sinkhorn marginals, eps->0 ≈ EMD, and GW on a toy heterogeneous pair (Spec §1.4, §1.6)."""
import numpy as np
import pytest

pot = pytest.importorskip("ot", reason="POT not installed")

from ot_lora_merge.ot_solve import sinkhorn, emd, entropic_gw, coupling


def _toy(r=4, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random(r); p /= p.sum()
    q = rng.random(r); q /= q.sum()
    C = rng.random((r, r))
    return p, q, C


def test_sinkhorn_marginals():
    p, q, C = _toy(5, seed=1)
    P = sinkhorn(p, q, C, eps=1e-2)
    assert np.allclose(P.sum(axis=1), p, atol=1e-4)
    assert np.allclose(P.sum(axis=0), q, atol=1e-4)
    assert np.all(P >= -1e-12)


def test_sinkhorn_approaches_emd():
    p, q, C = _toy(6, seed=2)
    P_exact = emd(p, q, C)
    P_eps = sinkhorn(p, q, C, eps=1e-3)
    cost_exact = float((P_exact * C).sum())
    cost_eps = float((P_eps * C).sum())
    # entropic transport cost is close to exact as eps -> 0
    assert abs(cost_eps - cost_exact) < 5e-2


def test_coupling_dispatch_exact():
    p, q, C = _toy(4, seed=3)
    P = coupling(p, q, C, eps=1e-2, exact=True)
    assert np.allclose(P.sum(axis=1), p, atol=1e-9)


def test_gw_heterogeneous_ranks():
    rng = np.random.default_rng(4)
    r1, r2 = 5, 3                       # different ranks
    D1 = rng.random((r1, r1)); D1 = (D1 + D1.T) / 2; np.fill_diagonal(D1, 0)
    D2 = rng.random((r2, r2)); D2 = (D2 + D2.T) / 2; np.fill_diagonal(D2, 0)
    p = np.full(r1, 1.0 / r1)
    q = np.full(r2, 1.0 / r2)
    P = entropic_gw(D1, D2, p, q, eps=5e-2)
    assert P.shape == (r1, r2)
    assert np.allclose(P.sum(axis=1), p, atol=1e-3)
    assert np.allclose(P.sum(axis=0), q, atol=1e-3)
