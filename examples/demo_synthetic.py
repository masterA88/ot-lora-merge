"""End-to-end demo of OT-LoRA-Merge on synthetic multi-task LoRA adapters — CPU only, no GPU/data.

Run:  python examples/demo_synthetic.py
Shows: (1) building task adapters with partially overlapping subspaces, (2) merging via align /
barycenter / heterogeneous-rank GW, (3) the subspace-overlap diagnostic. Sanity check that the
method runs and produces low-rank merged factors — the real benchmark is FusionBench (see experiments/).
"""
from __future__ import annotations

import numpy as np

from ot_lora_merge import extract_directions, merge_layer
from ot_lora_merge.metrics import subspace_overlap


def make_task_adapter(m, n, r, shared_basis_u, shared_basis_v, shared_frac=0.5, seed=0):
    """Build a LoRA adapter whose subspace partially overlaps a shared basis."""
    rng = np.random.default_rng(seed)
    k_shared = int(round(r * shared_frac))
    U = np.empty((m, r)); V = np.empty((n, r))
    U[:, :k_shared] = shared_basis_u[:, :k_shared]
    V[:, :k_shared] = shared_basis_v[:, :k_shared]
    U[:, k_shared:] = np.linalg.qr(rng.standard_normal((m, r - k_shared)))[0]
    V[:, k_shared:] = np.linalg.qr(rng.standard_normal((n, r - k_shared)))[0]
    U, _ = np.linalg.qr(U); V, _ = np.linalg.qr(V)
    sigma = np.sort(rng.uniform(0.5, 3.0, size=r))[::-1]
    dW = (U * sigma) @ V.T
    # factor to LoRA form B@A
    Uu, s, Vt = np.linalg.svd(dW, full_matrices=False)
    B = Uu[:, :r] * np.sqrt(s[:r]); A = np.sqrt(s[:r])[:, None] * Vt[:r, :]
    return {"A": A, "B": B, "scale": 1.0}


def main():
    rng = np.random.default_rng(42)
    m, n, r, T = 32, 24, 6, 4
    shared_u = np.linalg.qr(rng.standard_normal((m, r)))[0]
    shared_v = np.linalg.qr(rng.standard_normal((n, r)))[0]

    adapters = [
        make_task_adapter(m, n, r, shared_u, shared_v, shared_frac=0.5, seed=t)
        for t in range(T)
    ]

    print(f"{T} task adapters, rank r={r}, dims {m}x{n}\n")

    # pairwise subspace overlap (the 'overlap' axis of the overlap-vs-gain diagnostic)
    dirs = [extract_directions(a["A"], a["B"], scale=a["scale"]) for a in adapters]
    print("pairwise left-subspace overlap:")
    for i in range(T):
        row = "  " + " ".join(f"{subspace_overlap(dirs[i][0], dirs[j][0]):.2f}" for j in range(T))
        print(row)
    print()

    for mode in ("align", "barycenter"):
        A_star, B_star = merge_layer(adapters, mode=mode, exact=True)
        dW = B_star @ A_star
        eff_rank = int(np.linalg.matrix_rank(dW, tol=1e-8))
        print(f"[{mode:10s}] merged dW: shape={dW.shape}, ||dW||_F={np.linalg.norm(dW):.3f}, "
              f"effective rank={eff_rank}, A*={A_star.shape}, B*={B_star.shape}")

    # heterogeneous-rank merge via Gromov-Wasserstein (the structural moat)
    hetero = [
        make_task_adapter(m, n, 6, shared_u, shared_v, 0.5, seed=10),
        make_task_adapter(m, n, 3, shared_u[:, :3], shared_v[:, :3], 0.5, seed=11),
        make_task_adapter(m, n, 4, shared_u[:, :4], shared_v[:, :4], 0.5, seed=12),
    ]
    A_star, B_star = merge_layer(hetero, mode="gw", target_rank=6)
    print(f"\n[gw hetero ] merged adapters of ranks {[a['A'].shape[0] for a in hetero]} -> "
          f"A*={A_star.shape}, B*={B_star.shape}  "
          f"(Core-Space / GeoMerge structurally cannot do this)")


if __name__ == "__main__":
    main()
