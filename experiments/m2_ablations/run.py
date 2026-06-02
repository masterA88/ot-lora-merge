"""M2 — ablation grid: rank x eps x cost-variant x {align,barycenter} x GW. Spec §6.

Each cell = one CPU merge + one 8-task GPU eval. Sweeps are eval-bound; the OT merges are seconds.
Writes results/m2_ablations.csv (one row per cell). Use --dry-run to print the grid without eval.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RANKS = [8, 16, 32, 64]
EPS = ["exact", 1e-3, 1e-2, 1e-1, 1.0]
COSTS = ["paired", "left", "right"]
MODES = ["align", "barycenter"]


def grid():
    for r, eps, cost, mode in itertools.product(RANKS, EPS, COSTS, MODES):
        yield {"rank": r, "eps": eps, "cost": cost, "mode": mode,
               "exact": eps == "exact"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/modelpool/clip_vit_b32_knots_8task_lora.yaml")
    ap.add_argument("--out", default="results/m2_ablations.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = list(grid())
    print(f"{len(cells)} ablation cells")
    if args.dry_run:
        for c in cells[:10]:
            print(" ", c)
        print("  ...")
        return

    from experiments._eval import evaluate, load_modelpool
    from ot_lora_merge.fusionbench_hook import OTLoRAMergeAlgorithm

    modelpool = load_modelpool(args.config)
    specialist = evaluate(None, modelpool)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for c in cells:
        algo = OTLoRAMergeAlgorithm(mode=c["mode"], cost=c["cost"],
                                    eps=(0.0 if c["exact"] else c["eps"]),
                                    exact=c["exact"], target_rank=c["rank"])
        merged = algo.run(modelpool)
        accs = evaluate(merged, modelpool)
        avg = sum(accs[t] / specialist[t] for t in accs) / len(accs)
        rows.append({**c, "avg_norm_acc": avg})
        print(f"{c} -> {avg:.4f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
