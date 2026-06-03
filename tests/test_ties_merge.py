"""Unit tests for the OT-TIES merge (ties_merge.py)."""
import numpy as np

from ot_lora_merge.ties_merge import ot_ties_layer, ot_ties_model, ties_combine


def _adapter(m, n, r, seed):
    rng = np.random.default_rng(seed)
    return {"A": rng.standard_normal((r, n)), "B": rng.standard_normal((m, r)), "scale": 1.0}


def test_ties_combine_sign_election():
    # 3 tasks, k=1, n=2. col0: 2,4,-3 -> sum=+3 -> elected +, mean over the agreeing + tasks (2,4)=3.
    # col1: all +1 -> mean 1.
    c = [np.array([[2.0, 1.0]]), np.array([[4.0, 1.0]]), np.array([[-3.0, 1.0]])]
    out = ties_combine(c, keep=1.0, lam=1.0)
    assert out[0, 0] == 3.0           # mean(2,4) over agreeing-with-sum (+) tasks; -3 excluded
    assert np.isclose(out[0, 1], 1.0)


def test_ties_combine_trim_keeps_top_fraction():
    c = [np.array([[10.0, 0.1, 0.1, 0.1]])]
    out = ties_combine(c, keep=0.25, lam=1.0)   # keep only the largest entry
    assert out[0, 0] == 10.0
    assert np.allclose(out[0, 1:], 0.0)


def test_ot_ties_layer_shape_and_finite():
    m, n, r = 32, 24, 4
    adapters = [_adapter(m, n, r, s) for s in range(3)]
    dW = ot_ties_layer(adapters, k=8, keep=0.5, lam=0.7)
    assert dW.shape == (m, n)
    assert np.isfinite(dW).all()


def test_ot_ties_single_adapter_identity():
    # one adapter -> merged update == its own dW (identity short-circuit, no transport)
    ad = _adapter(16, 12, 3, 7)
    dW = ot_ties_layer([ad], k=8)
    expected = ad["scale"] * (ad["B"] @ ad["A"])
    assert np.allclose(dW, expected)


def test_ot_ties_model_keys():
    layer_adapters = {
        "l0": [_adapter(16, 12, 3, s) for s in range(2)],
        "l1": [_adapter(16, 12, 3, s + 10) for s in range(2)],
    }
    out = ot_ties_model(layer_adapters, k=6, keep=0.5, lam=0.7)
    assert set(out) == {"l0", "l1"}
    assert out["l0"].shape == (16, 12)
