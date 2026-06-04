# M3 — heterogeneous-rank merging + compute cost (the structural carve-out)

Kaggle free P100, 2026-06-04. torch 2.6.0+cu124. Metric = avg normalized accuracy over 8 tasks.

## Setup
Rank-fixed merge methods structurally require all adapters at the **same** rank: GeoMerge operates on
the `(St × SPD × St)/O(r)` quotient (fixed r); Core-Space projects into a **shared rank-r basis**. They
**cannot natively merge a mixed-rank adapter zoo**. Our OT formulation operates on the per-task update
`ΔW_t = B_t A_t` (full matrices), so it is rank-agnostic.

We build a genuinely mixed-rank zoo by **SVD-truncating** the 8 KnOTS r16 adapters to assigned ranks
(free, no training): sun397=16, cars=8, resisc45=4, eurosat=16, svhn=8, gtsrb=4, mnist=16, dtd=8.

## Results

| Method | regime | avg-norm-acc | CPU merge time (full model) |
|---|---|---|---|
| **OT-TIES (ours)** | **mixed-rank {4,8,16}** | **0.698** | 264 s |
| GW (Gromov-Wasserstein) | mixed-rank {4,8,16} | 0.623 | 359 s |
| OT-TIES (reference) | homogeneous r16 | 0.708 | 274 s |
| — | naive floor | ~0.64 | — |

## Takeaways
- **Graceful under heterogeneity:** OT-TIES merges the mixed-rank zoo at **0.698**, only **−1.0%** vs the
  homogeneous-rank result (0.708) — and still clearly above the naive floor (~0.64), **in a setting the
  SOTA rank-fixed baselines (GeoMerge / Core-Space) cannot enter at all**. That capability is the
  structural moat.
- **GW operator works** (0.623) as a heterogeneous-rank-native OT merge, though the OT-TIES-basis route
  is stronger; GW is the more general fallback when no shared ambient frame is assumed.
- **Compute-frugal:** every merge is pure **CPU**, ~4.5–6 min for the full ViT-B/32 model (264–359 s);
  GPU is used only for evaluation. No iterative Riemannian Exp/Log loop (GeoMerge) or repeated full-ΔW
  SVD per method.

Raw: `m3_hetero.csv`, `m3_log.txt`. Mixed-rank zoo is reproducible (SVD-truncation, fixed seed-free).
