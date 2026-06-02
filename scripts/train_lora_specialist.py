"""Fallback: train ONE LoRA specialist on ONE task (CLIP-ViT-B/32, base frozen). <1 GPU-hr on Kaggle T4.

Only needed if pre-released adapters for a given rank are not on the HF hub. Prefer
scripts/download_adapters.py first. Caches to adapters_cache/<task>_r<rank>/.

Usage (Kaggle/Colab):
    python scripts/train_lora_specialist.py --task eurosat --rank 16 --epochs 5
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--base", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default="adapters_cache")
    args = ap.parse_args()

    # Implementation outline (wired against the installed stack in M0):
    #   1. load CLIP image encoder (transformers/open_clip), FREEZE base weights
    #   2. attach PEFT LoraConfig(r=args.rank, target_modules=attn/mlp proj, lora_alpha=2*r)
    #   3. load the task dataset via torchvision / HF datasets (see configs/modelpool)
    #   4. train a linear classification head + LoRA for args.epochs (AdamW, cosine)
    #   5. save adapter to {out}/{task}_r{rank}/  (PEFT save_pretrained) + record specialist acc
    raise SystemExit(
        "Training stub: finalize against the installed PEFT/transformers stack in M0. "
        "Base model is frozen, so each run is <1 GPU-hr on a free T4. "
        "Check scripts/download_adapters.py for pre-released adapters first."
    )


if __name__ == "__main__":
    main()
