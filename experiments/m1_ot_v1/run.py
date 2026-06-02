"""M1 — first OT-LoRA-Merge head-to-head (align + barycenter) vs the M0 baselines.

The merge runs on CPU via ot_lora_merge; FusionBench provides load + eval. Writes results/m1_ot_v1.csv.

Usage:  python experiments/m1_ot_v1/run.py --config configs/modelpool/clip_vit_b32_knots_8task_lora.yaml \
                                           --method configs/method/ot_lora_barycenter.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiments._eval import evaluate, load_modelpool  # noqa: E402
from ot_lora_merge.fusionbench_hook import OTLoRAMergeAlgorithm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", default="results/m1_ot_v1.csv")
    args = ap.parse_args()

    with open(args.method) as f:
        mcfg = {k: v for k, v in yaml.safe_load(f).items() if k != "name"}

    modelpool = load_modelpool(args.config)
    specialist = evaluate(None, modelpool)

    algo = OTLoRAMergeAlgorithm(**{k: mcfg[k] for k in mcfg
                                   if k in ("mode", "cost", "eps", "exact", "target_rank", "gw_eps")})
    merged = algo.run(modelpool)
    accs = evaluate(merged, modelpool)
    norm = [accs[t] / specialist[t] for t in accs]
    avg = sum(norm) / len(norm)
    print(f"OT-LoRA-Merge [{mcfg.get('mode')}] avg-norm-acc = {avg:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        row = {"method": f"ot_lora_{mcfg.get('mode')}", "avg_norm_acc": avg, **accs}
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
