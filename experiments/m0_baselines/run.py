"""M0 — reproduce the published baselines (grant credibility).

Runs FusionBench's built-in merges (simple-average, task-arithmetic, ties, adamerging,
knots, core-space) on the 8-task ViT-B/32 LoRA pool and writes a baseline table to
results/m0_baselines.csv. Target: reproduce published numbers within ~1%.
Merges are CPU; only eval uses the GPU.

Usage:
    python experiments/m0_baselines/run.py \
        --config configs/modelpool/clip_vit_b32_knots_8task_lora.yaml \
        [--taskpool configs/taskpool/clip_vit_b32_8task.yaml] \
        [--out results/m0_baselines.csv]

FusionBench algorithm names used here (sourced from fusion_bench-0.2.32 sdist):

  Modern v0.2.x API: algorithms are loaded via _target_ in a YAML config or directly
  by importing their classes. The class names below are confirmed from:
    fusion_bench-0.2.32/fusion_bench/method/__init__.py

  simple_average   -> fusion_bench.method.simple_average.SimpleAverageAlgorithm
    Source: fusion_bench-0.2.32/fusion_bench/method/simple_average.py

  task_arithmetic  -> fusion_bench.method.task_arithmetic.TaskArithmeticAlgorithm
    Source: fusion_bench-0.2.32/fusion_bench/method/task_arithmetic/ (inferred from __init__)
    # VERIFY-IN-KAGGLE: import from fusion_bench.method import TaskArithmeticAlgorithm

  ties             -> fusion_bench.method.ties_merging.TiesMergingAlgorithm
    Source: fusion_bench-0.2.32/fusion_bench/method/__init__.py _import_structure

  adamerging       -> fusion_bench.method.adamerging.CLIPTaskWiseAdaMergingAlgorithm
    Source: fusion_bench-0.2.32/fusion_bench/method/__init__.py _import_structure
    # VERIFY-IN-KAGGLE: task-wise vs layer-wise. CLIPTaskWiseAdaMergingAlgorithm is the
    # lighter variant; use CLIPLayerWiseAdaMergingAlgorithm for the published numbers.

  knots            -> Not present in fusion_bench-0.2.32 method registry.
    The KnOTS paper (2410.19735) has a separate repo: apdoublezz/KnOTS.
    # VERIFY-IN-KAGGLE: check if `from fusion_bench.method import TaskSingularVectorMerging`
    # is the FusionBench equivalent of KnOTS (it uses SVD in ΔW space similarly).
    # TaskSingularVectorMerging is in fusion_bench-0.2.32/fusion_bench/method/__init__.py.

  core_space       -> fusion_bench.method.isotropic_merging.IsotropicMergingInCommonSubspace
    (ISO-C in the Core Space paper; "ISO_C_Merge" alias available)
    Source: fusion_bench-0.2.32/fusion_bench/method/isotropic_merging/__init__.py
    # VERIFY-IN-KAGGLE: also check IsotropicMergingInCommonAndTaskSubspace (ISO-CTS) for
    # the full Core Space best number.

  NOTE on compat v0.1.x AlgorithmFactory.create_algorithm():
    The DEPRECATED name-based registry (fusion_bench.compat.method.AlgorithmFactory) does NOT
    contain simple_average, task_arithmetic, ties, or core_space directly by those short names.
    Do NOT use load_algorithm_from_name() — it doesn't exist in v0.2.x.
    Use the direct class import pattern below.
    Source: fusion_bench-0.2.32/fusion_bench/compat/method/__init__.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiments._eval import evaluate, load_modelpool, set_seed  # noqa: E402

log = logging.getLogger(__name__)

# Map from short baseline name to FusionBench class import path.
# Confirmed from fusion_bench-0.2.32/fusion_bench/method/__init__.py _import_structure.
# VERIFY-IN-KAGGLE: import each class and assert issubclass(cls, BaseAlgorithm).
BASELINE_CLASSES = {
    "simple_average": "fusion_bench.method.simple_average.SimpleAverageAlgorithm",
    "task_arithmetic": "fusion_bench.method.task_arithmetic.TaskArithmeticAlgorithm",
    "ties": "fusion_bench.method.ties_merging.TiesMergingAlgorithm",
    # AdaMerging: task-wise is lighter; layer-wise matches published numbers.
    # VERIFY-IN-KAGGLE: use CLIPLayerWiseAdaMergingAlgorithm for the published number.
    "adamerging": "fusion_bench.method.adamerging.CLIPTaskWiseAdaMergingAlgorithm",
    # KnOTS equivalent in FusionBench: TaskSingularVectorMerging (SVD in ΔW space).
    # VERIFY-IN-KAGGLE: confirm this is the right stand-in, or install KnOTS separately.
    "knots": "fusion_bench.method.task_singular_vector.TaskSingularVectorMerging",
    # Core Space (ISO-C): isotropic merging in common subspace.
    "core_space": "fusion_bench.method.isotropic_merging.IsotropicMergingInCommonSubspace",
}

# Default constructor kwargs per baseline (all optional; empty = defaults from class)
BASELINE_KWARGS: dict = {
    "simple_average": {},
    "task_arithmetic": {},   # VERIFY-IN-KAGGLE: may need scaling_factor param
    "ties": {},
    "adamerging": {},
    "knots": {},
    "core_space": {},
}


def _import_class(dotted_path: str):
    """Import a class from a dotted module path string."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Cannot parse class path: {dotted_path!r}")
    module_path, class_name = parts
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def run_baseline(name: str, modelpool, kwargs: dict | None = None):
    """Instantiate a FusionBench algorithm by class path and run it on the modelpool.

    This is the corrected implementation of the stub's load_algorithm_from_name(),
    which does not exist in FusionBench v0.2.x. The modern pattern is direct class
    import + instantiation, or from_yaml() for YAML-driven configs.

    Source: fusion_bench-0.2.32/fusion_bench/method/base_algorithm.py run() contract.
    # VERIFY-IN-KAGGLE: if a baseline fails with TypeError on __init__, inspect the
    # class signature and add the required kwargs to BASELINE_KWARGS above.
    """
    if name not in BASELINE_CLASSES:
        raise ValueError(f"Unknown baseline {name!r}. Available: {list(BASELINE_CLASSES)}")

    cls = _import_class(BASELINE_CLASSES[name])
    kw = kwargs if kwargs is not None else BASELINE_KWARGS.get(name, {})
    algo = cls(**kw)
    return algo.run(modelpool)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="M0: reproduce baseline merging results")
    ap.add_argument("--config", required=True,
                    help="Path to modelpool YAML (configs/modelpool/clip_vit_b32_knots_8task_lora.yaml)")
    ap.add_argument("--baselines", nargs="+", default=list(BASELINE_CLASSES.keys()),
                    help="Which baselines to run (default: all)")
    ap.add_argument("--out", default="results/m0_baselines.csv")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    set_seed(args.seed)

    modelpool = load_modelpool(args.config)
    log.info("Loaded modelpool with tasks: %s", modelpool.model_names)

    # Specialist accuracies (evaluate each task model individually)
    log.info("Evaluating specialist (per-task) accuracies …")
    specialist = evaluate(None, modelpool)
    log.info("Specialist: %s", specialist)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []

    for name in args.baselines:
        log.info("Running baseline: %s", name)
        try:
            merged = run_baseline(name, modelpool)
        except Exception as exc:
            log.error("Baseline %s failed: %s", name, exc)
            continue

        accs = evaluate(merged, modelpool)
        # Compute per-task normalized accuracy
        norm_accs = {
            t: accs[t] / specialist[t]
            for t in accs
            if t != "average" and t in specialist and specialist[t] > 0
        }
        avg_norm = sum(norm_accs.values()) / len(norm_accs) if norm_accs else 0.0
        row = {"method": name, "avg_norm_acc": avg_norm, **accs}
        rows.append(row)
        log.info("%s  avg-norm-acc=%.4f  avg-acc=%.4f", name, avg_norm, accs.get("average", 0.0))

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s", args.out)
    else:
        log.warning("No results to write.")


if __name__ == "__main__":
    main()
