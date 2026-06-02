"""Identity guarantees: T=1 == identity; T copies of one adapter == that adapter (Spec §1.7)."""
import numpy as np
import pytest

pytest.importorskip("ot", reason="POT not installed")

from ot_lora_merge.directions import extract_directions, reconstruct
from ot_lora_merge.merge import merge_layer


def _adapter(m, n, r, seed):
    rng = np.random.default_rng(seed)
    return {"A": rng.standard_normal((r, n)), "B": rng.standard_normal((m, r)), "scale": 1.0}


def _dW(adapter):
    U, s, V, _ = extract_directions(adapter["A"], adapter["B"], scale=adapter.get("scale"))
    return reconstruct(U, s, V)


def test_single_adapter_is_identity():
    ad = _adapter(10, 8, 4, seed=1)
    A_star, B_star = merge_layer([ad], mode="align")
    assert np.allclose(B_star @ A_star, _dW(ad), atol=1e-8)


@pytest.mark.parametrize("mode", ["align", "barycenter"])
def test_identical_copies_recover_adapter(mode):
    ad = _adapter(12, 9, 4, seed=2)
    copies = [dict(ad), dict(ad), dict(ad)]
    A_star, B_star = merge_layer(copies, mode=mode, exact=True)
    assert np.allclose(B_star @ A_star, _dW(ad), atol=1e-6)


def test_align_beats_naive_on_aligned_subspace():
    # Two adapters sharing the same subspace but different magnitudes: merge stays in-subspace.
    rng = np.random.default_rng(3)
    m, n, r = 12, 10, 3
    Q_u, _ = np.linalg.qr(rng.standard_normal((m, r)))
    Q_v, _ = np.linalg.qr(rng.standard_normal((n, r)))
    ad1 = {"A": (np.diag([3.0, 2.0, 1.0]) @ Q_v.T), "B": Q_u, "scale": 1.0}
    ad2 = {"A": (np.diag([1.0, 2.0, 3.0]) @ Q_v.T), "B": Q_u, "scale": 1.0}
    A_star, B_star = merge_layer([ad1, ad2], mode="align", exact=True)
    dW = B_star @ A_star
    # merged update must live in span(Q_u) x span(Q_v)
    resid = dW - Q_u @ (Q_u.T @ dW @ Q_v) @ Q_v.T
    assert np.linalg.norm(resid) < 1e-7


def test_heterogeneous_rank_gw_runs():
    ad_r4 = _adapter(12, 9, 4, seed=5)
    ad_r2 = _adapter(12, 9, 2, seed=6)
    A_star, B_star = merge_layer([ad_r4, ad_r2], mode="gw", target_rank=4)
    assert A_star.shape[0] <= 4
    assert B_star.shape[0] == 12
    assert np.all(np.isfinite(A_star)) and np.all(np.isfinite(B_star))
