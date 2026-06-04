# OT-LoRA-Merge

Merge a bank of task-specialized LoRA adapters by solving a small **optimal-transport (Sinkhorn)
coupling / Wasserstein barycenter in the rank-r subspace** of each adapter — with a
**Gromov-Wasserstein** variant that natively merges **heterogeneous-rank** adapters, a setting the
SVD and Riemannian baselines structurally cannot enter.

> One line: treat each task's LoRA update `ΔW = BA` as an empirical distribution over its rank-r
> singular directions, weighted by singular value, and transport/average those distributions.

## Why this is interesting

- The OT solve is **r × r** (r ≈ 8–64) → microseconds–milliseconds on **CPU**. No GPU needed for the
  merge itself; GPU is only used to *evaluate* the merged model. Compute-frugal merging is itself a
  contribution (model reuse for low-resource labs).
- Closest prior art and how we differ:
  - **Core Space** (NeurIPS'25): deterministic *lossless SVD projection* into a shared basis, then sum.
    We replace the hard projection with a **soft OT coupling** — same r×r cheapness, soft many-to-many.
  - **GeoMerge / Fréchet averages** (2026): one *Riemannian geodesic* per task on the shared-rank
    `(St×SPD×St)/O(r)` quotient. We use a **transport barycenter between direction distributions** —
    and, via Gromov-Wasserstein, merge **unequal ranks** natively.
  - **OTMF** (2511.19561): OT between *feature/activation distributions* to learn masks (data-dependent,
    full-model). Ours is OT between *rank-r factor directions* as the **weight-only** merge operator —
    no data forward pass for the merge.

## Bar to beat (8-task CLIP-ViT-B/32 LoRA, average normalized accuracy)

Benchmark = the **KnOTS-lineage ViT-B/32 rank-16 adapters** (`hoffman-lab/KnOTS-ViT-B-32_lora_R16_<task>`),
the exact specialists on which GeoMerge / Core-Space / KnOTS report — so these numbers are a
**direct, apples-to-apples** comparison. Adapter facts verified via `scripts/kaggle_probe.py` (2026-06-01):
r=16, alpha=16 (scale 1.0), target `[q,k,v,out]_proj`, base `openai/clip-vit-base-patch32`.
(FusionBench's *bundled* LoRA pool is B/16 — a different backbone — so we use FusionBench only for the
eval harness + built-in baselines and load the KnOTS B/32 adapters into it.)

| Method | avg-norm-acc (8-task ViT-B/32 LoRA) |
|---|---|
| GeoMerge (reported) | 77.10% |
| Core Space (reported) | ~76.43% |
| KnOTS-TIES (reported, full method) | 74.02% |
| **OT-TIES (ours, λ=0.7 keep=0.2)** | **70.79%** |
| svd-TIES (our KnOTS-style control, left-basis) | 68.7% |
| **— naive baseline floor (reproduced by us, M0) —** | |
| Task-Arithmetic (α=0.1) | 63.9% (ours) ≈ 63.78% published ✓ |
| Simple average | 63.6% (ours) |
| TIES (disjoint-mean, α=0.3) | 63.4% (ours) |

**Result (M1–M2, see `results/m2_results.md`).** Merging by OT *averaging* (Wasserstein barycenter)
plateaus at the naive floor — averaging blurs, it does not resolve sign interference. The win comes
from **TIES interference-resolution inside a shared subspace**, and our contribution is that an
**OT-chosen subspace beats the SVD-chosen one (KnOTS-style) under identical TIES, at every scaling**
(+0.7…+1.9%). M0 reproduces published Task-Arithmetic within 0.1% (harness validated).

## Status

Research repo. CPU OT core + the headline **OT-TIES** merge (`ot_lora_merge.ot_ties_model`) implemented
and unit-tested (17/17). GPU eval harness wraps **FusionBench** (apples-to-apples baselines + eval).
**M0–M3 complete** (all on free Kaggle GPU; see `results/m0_results.md`, `m2_results.md`, `m3_results.md`):
- **M0** baselines reproduced (Task-Arithmetic within 0.1% of published) — harness validated.
- **M1–M2** OT-TIES: an **OT-chosen merge subspace beats the SVD-chosen one (KnOTS-style) under identical
  TIES, at every scaling** (+0.7…+1.9%); peak **0.708** vs control 0.687, floor 0.64.
- **M3** heterogeneous-rank carve-out: OT-TIES merges a mixed-rank {4,8,16} adapter zoo at **0.698**
  (−1.0% vs homogeneous) — **a setting GeoMerge / Core-Space structurally cannot enter**. All merges are
  pure CPU (~4–6 min/model).

Remaining: write-up (workshop paper); optional accuracy push toward the full-KnOTS 0.74.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
pytest -q                   # runs the CPU OT-core tests (no GPU/data needed)
```

## Repro (free compute)

The full results table reproduces on **Kaggle (T4/P100, 30 GPU-hr/week)** + **Colab free** + CPU OT
solve. See `env/kaggle-notebook.ipynb`. Merges run on CPU; GPU is only for the 8-task eval.

## Milestones

- **M0** `experiments/m0_baselines` — reproduce TA / TIES / AdaMerging / KnOTS / Core-Space (grant credibility).
- **M1** `experiments/m1_ot_v1` — first OT-LoRA-Merge head-to-head (align + barycenter).
- **M2** `experiments/m2_ablations` — rank × ε × cost-variant × ±barycenter × GW grid.
- **M3** `experiments/m3_carveout` — heterogeneous-rank GW + merge-time benchmark (the structural moat).

## Layout

```
src/ot_lora_merge/   directions · cost · ot_solve · align · barycenter · merge · metrics  (the method)
configs/             FusionBench-compatible method + baseline + ablation configs
experiments/         one folder per milestone (one editable surface + one metric + a budget)
scripts/             train/download LoRA specialists; build results table
tests/               CPU unit tests for the OT core
env/                 Kaggle / Colab free-tier runners
```

## License

MIT. All datasets, models, and libraries are free / open-source. See `experiment-spec`-style audit in
the research workspace.
