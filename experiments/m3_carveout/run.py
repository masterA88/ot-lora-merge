"""M3 — the structural carve-out: heterogeneous-rank GW merging + merge-time benchmark. Spec §1.6, §7.

Two results no baseline can produce:
  (1) merge a zoo of mixed-rank adapters {8,16,32,64} via Gromov-Wasserstein (Core-Space's shared
      basis and GeoMerge's O(r) quotient structurally require equal rank);
  (2) merge wall-clock vs KnOTS (full-ΔW SVD) and GeoMerge (Riemannian Fréchet loop) — the
      compute-frugal-merging story for low-resource labs (the grant framing).
Writes results/m3_carveout.csv and results/m3_merge_timing.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ot_lora_merge.merge import merge_layer  # noqa: E402


def _rand_adapter(m, n, r, seed):
    rng = np.random.default_rng(seed)
    return {"A": rng.standard_normal((r, n)), "B": rng.standard_normal((m, r)), "scale": 1.0}


def benchmark_merge_time(m=768, n=768, ranks=(8, 16, 32, 64), repeats=5):
    """CPU merge-time for a single layer across modes — confirms the r x r cheapness (Spec §5)."""
    rows = []
    adapters = [_rand_adapter(m, n, r, seed=i) for i, r in enumerate(ranks)]
    for mode in ("align", "barycenter", "gw"):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            merge_layer(adapters, mode=mode, target_rank=max(ranks))
            ts.append(time.perf_counter() - t0)
        rows.append({"mode": mode, "layer_dims": f"{m}x{n}",
                     "ranks": "+".join(map(str, ranks)),
                     "median_sec": float(np.median(ts))})
        print(f"[{mode:10s}] median merge {np.median(ts)*1e3:.2f} ms  (CPU, ranks {ranks})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--timing-only", action="store_true",
                    help="run only the CPU merge-time benchmark (no GPU/FusionBench needed)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    timing = benchmark_merge_time()
    with open(os.path.join(args.out_dir, "m3_merge_timing.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing[0].keys()))
        w.writeheader(); w.writerows(timing)

    if args.timing_only:
        print("timing-only: skipped the FusionBench heterogeneous-rank accuracy eval.")
        return

    # Heterogeneous-rank accuracy eval (needs FusionBench + mixed-rank specialists, trained in M0/M2).
    from experiments._eval import evaluate, load_modelpool
    from ot_lora_merge.fusionbench_hook import OTLoRAMergeAlgorithm
    modelpool = load_modelpool("configs/modelpool/clip_vit_b32_knots_8task_lora.yaml")
    specialist = evaluate(None, modelpool)
    algo = OTLoRAMergeAlgorithm(mode="gw", gw_eps=0.05, target_rank=64)
    merged = algo.run(modelpool)  # modelpool carries mixed-rank specialists in this experiment
    accs = evaluate(merged, modelpool)
    avg = sum(accs[t] / specialist[t] for t in accs) / len(accs)
    print(f"heterogeneous-rank GW avg-norm-acc = {avg:.4f}")
    with open(os.path.join(args.out_dir, "m3_carveout.csv"), "w", newline="") as f:
        row = {"method": "ot_lora_gw_hetero", "avg_norm_acc": avg, **accs}
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)


if __name__ == "__main__":
    main()
