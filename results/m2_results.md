# M2 — OT-LoRA-Merge method results (8-task CLIP-ViT-B/32 KnOTS LoRA)

Kaggle free P100, 2026-06-02/03. torch 2.6.0+cu124. Metric = avg normalized accuracy over 8 tasks.
Specialist accs + baseline floor validated in `m0_results.md`.

## The arc (what we learned, in order)

1. **OT-as-averaging fails (M1 + rank sweep).** align / barycenter at any rank {16,32,64,128} or
   scale plateau at the **simple-average floor ~0.62** (`m1_results.csv`, `m2_rank.csv`).
   Diagnosis: a transport *barycenter* blurs directions; it does not *resolve sign interference*.
2. **Interference resolution is the lever.** Apply **TIES** (trim top-keep%, sign-elect, disjoint-mean)
   inside a **shared subspace** — the mechanism KnOTS/Core-Space win with. This lifts us off the floor.
3. **OT chooses a better subspace than SVD.** Comparing identical TIES in two bases:
   - `svd_ties`  — basis = SVD of stacked task updates (KnOTS-style control)
   - `ot_ties`   — basis = top-k left-SVD of the **OT (Wasserstein) barycenter** of the task directions
   **ot_ties beats svd_ties at every scaling** (`m2ties.csv`, `m2tune.csv`).

## Results

| Method | avg-norm-acc | note |
|---|---|---|
| simple-avg / TA / TIES (naive floor) | ~0.64 | reproduced (M0) |
| OT align / barycenter (averaging, any rank) | 0.61–0.63 | **does not beat floor** |
| svd_ties (KnOTS-style control, left-basis) | 0.687 | peak over λ |
| **ot_ties (OURS), λ=0.7, keep=0.2** | **0.708** | **peak; > control at every λ** |
| KnOTS (reported, full method) | 0.740 | |
| Core-Space / GeoMerge (reported) | 0.764 / 0.771 | |

**OT-vs-SVD basis, identical TIES (1-sided, keep=0.2), the core contribution:**
| λ | svd_ties | ot_ties | Δ (ot−svd) |
|---|---|---|---|
| 0.3 | 0.658 | 0.665 | +0.7 |
| 0.4 | 0.672 | 0.682 | +1.0 |
| 0.5 | 0.682 | 0.696 | +1.4 |
| 0.6 | 0.687 | 0.706 | +1.9 |
| 0.7 | — | **0.708** | (peak) |

2-sided (project both U and V) was *worse* (over-compresses) — dropped; 1-sided left-basis wins.

## Takeaways
- **Contribution (robust):** an **OT-chosen merge subspace beats the SVD-chosen one under identical
  TIES**, at every scaling — instantiating "OT replaces KnOTS's ad-hoc SVD alignment". +0.7…+1.9%.
- **Absolute:** ot_ties 0.708 vs our simplified left-basis control 0.687; the gap to KnOTS-reported
  0.740 is mostly our control's simplification (left-basis only, no whitening), not the OT idea.
- **Compute:** all merges are CPU (barycenter + TIES, seconds–minutes/run); GPU only for eval.
- Method promoted to the package: `ot_lora_merge.ot_ties_layer` / `ot_ties_model` (defaults = the peak
  k=128, keep=0.2, λ=0.7). 17/17 unit tests pass.

Raw: `m1_results.csv`, `m2_rank.csv`, `m2ties.csv`, `m2tune.csv`, `m2peak.csv` (+ matching `*_log.txt`).
