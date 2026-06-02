"""Pull pre-released 8-task ViT-B/32 LoRA specialists from the HF hub (cached). Spec §3.

Prefer pre-released adapters referenced by FusionBench / KnOTS / Core-Space modelpools — training
32 specialists (8 tasks x 4 ranks) is the single biggest free-tier cost, so download when possible.
Falls back to scripts/train_lora_specialist.py only for missing (task, rank) pairs.
"""
from __future__ import annotations

import argparse

TASKS = ["sun397", "stanford_cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default="adapters_cache")
    args = ap.parse_args()
    # huggingface_hub.snapshot_download for each task's published adapter repo; record source hash.
    # Concrete repo IDs pinned in M0 against the FusionBench/KnOTS/Core-Space released pools.
    raise SystemExit(
        "Download stub: pin the HF adapter repo IDs in M0 (from FusionBench/KnOTS/Core-Space "
        f"released ViT-B/32 r={args.rank} pools), then snapshot_download into {args.out}/."
    )


if __name__ == "__main__":
    main()
