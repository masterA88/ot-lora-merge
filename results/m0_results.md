# M0 — baseline reproduction (8-task CLIP-ViT-B/32 KnOTS LoRA)

Run on Kaggle (free P100), 2026-06-02. Kernel: `master888x/ot-lora-merge-m0-baselines-b32-knots-v2`
+ `master888x/ot-lora-merge-m0-baseline-tuning`. Env: torch 2.6.0+cu124, FusionBench (git),
`merge_and_unload: true` (baseline path). Adapters: `hoffman-lab/KnOTS-ViT-B-32_lora_R16_*`.
Headline metric = average normalized accuracy = merged_acc / specialist_acc, over the 8 tasks.

## Specialist (fine-tuned) accuracies — eval VALIDATED (match published KnOTS)
| sun397 | cars | resisc45 | eurosat | svhn | gtsrb | mnist | dtd |
|---|---|---|---|---|---|---|---|
| .649 | .741 | .886 | .996 | .965 | .930 | .994 | .580 |

## Baselines (avg normalized accuracy) — harness reproduces published
| Method | config | avg-norm-acc | note |
|---|---|---|---|
| Simple average | — | 0.636 | in-range |
| Task-Arithmetic | **α=0.1** | **0.639** | **≈ published TA-Full 63.78% (within 0.1%) → harness validated** |
| Task-Arithmetic | α=0.2 | 0.566 | |
| Task-Arithmetic | α=0.3 | 0.405 | over-scaled |
| Task-Arithmetic | α=0.4 | 0.239 | over-scaled |
| TIES | mean, α=0.3, thr=20 | 0.634 | in-range |
| TIES | mean, α=1.0, thr=20 | 0.454 | over-scaled |

**Key facts learned:**
- FusionBench TA **sums all 8 task vectors then scales by `scaling_factor`** → optimum is small α (~0.1);
  α=0.3 (FB default) over-scales for 8 interfering LoRA tasks. TIES likewise: disjoint-`mean` ≫ `sum`,
  small α best.
- All *naive* baselines cluster at **~0.63–0.64**. This is the expected literature pattern: the naive
  floor is ~64%; the **bar to beat is the advanced methods — KnOTS 74.0%, Core-Space 76.4%, GeoMerge 77.1%.**

## Status
- ✅ Eval pipeline validated (specialists + TA reproduce published).
- ⏳ Next: add KnOTS / Core-Space (the real comparison) and/or M1 = our OT-LoRA-Merge.
- Raw: `m0_baselines.csv` (initial), `m0_tune.csv` (sweep), `m0_log.txt`, `m0_tune_log.txt`.
