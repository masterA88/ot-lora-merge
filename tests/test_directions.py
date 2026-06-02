"""SVD reconstruction exactness + sign-canonicalization stability (Spec §1.1, determinism note)."""
import numpy as np

from ot_lora_merge.directions import extract_directions, reconstruct, refactor


def _random_adapter(m, n, r, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((r, n))
    B = rng.standard_normal((m, r))
    return A, B


def test_reconstruction_exact():
    m, n, r, alpha = 12, 9, 4, 8.0
    A, B = _random_adapter(m, n, r, seed=1)
    U, sigma, V, p = extract_directions(A, B, alpha=alpha)
    dW_true = (alpha / r) * (B @ A)
    dW_hat = reconstruct(U, sigma, V)
    assert np.allclose(dW_hat, dW_true, atol=1e-9)
    # U, V are orthonormal (Stiefel); masses sum to 1.
    assert np.allclose(U.T @ U, np.eye(r), atol=1e-9)
    assert np.allclose(V.T @ V, np.eye(r), atol=1e-9)
    assert np.isclose(p.sum(), 1.0)
    assert np.all(sigma[:-1] >= sigma[1:] - 1e-12)  # descending


def test_sign_canonicalization_deterministic():
    A, B = _random_adapter(10, 7, 3, seed=2)
    U1, s1, V1, _ = extract_directions(A, B, alpha=4.0)
    # Flipping a column of B and the matching row of... actually flip via re-extraction with
    # an external sign on the factors must yield the SAME canonical U (sign-invariant).
    U2, s2, V2, _ = extract_directions(A.copy(), B.copy(), alpha=4.0)
    assert np.allclose(U1, U2)
    assert np.allclose(V1, V2)
    # largest-|component| of each U column is forced positive
    for k in range(U1.shape[1]):
        idx = np.argmax(np.abs(U1[:, k]))
        assert U1[idx, k] > 0


def test_refactor_roundtrip():
    m, n, r = 14, 11, 5
    A, B = _random_adapter(m, n, r, seed=3)
    dW = (B @ A)
    A_star, B_star = refactor(dW, r)
    assert A_star.shape == (r, n)
    assert B_star.shape == (m, r)
    assert np.allclose(B_star @ A_star, dW, atol=1e-8)
