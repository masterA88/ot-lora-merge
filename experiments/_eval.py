"""Shared FusionBench eval helpers for the milestone runners.

The GPU-dependent boundary. The OT merge itself is CPU (ot_lora_merge core); FusionBench is
used only to (a) load the 8-task ViT-B/32 LoRA modelpool and (b) evaluate a merged model.
Kept in one place so M0–M3 share identical eval + the apples-to-apples baseline numbers.

FusionBench v0.2.x API used here (sourced from fusion_bench-0.2.32 sdist):

  Modelpool loading (modern v0.2.x pattern):
    fusion_bench.modelpool.BaseModelPool.from_yaml(config_path)
      Reads the YAML, dispatches to the _target_ class via OmegaConf/Hydra instantiate().
      The _target_ for TA8-lora configs is fusion_bench.modelpool.CLIPVisionModelPool.
    Source: fusion_bench-0.2.32/fusion_bench/mixins/serialization.py from_yaml()
    Source: fusion_bench-0.2.32/fusion_bench_config/modelpool/CLIPVisionModelPool/
             clip-vit-base-patch16_TA8_lora.yaml  (FusionBench's bundled LoRA pool — B/16 — used
             only as the YAML schema template; our actual pool is the KnOTS B/32 adapters in
             configs/modelpool/clip_vit_b32_knots_8task_lora.yaml)

  Compat modelpool loading (v0.1.x / type-key configs):
    fusion_bench.compat.modelpool.load_modelpool_from_config(DictConfig)
    Source: fusion_bench-0.2.32/fusion_bench/compat/modelpool/__init__.py

  Taskpool / eval (modern v0.2.x):
    fusion_bench.taskpool.clip_vision.CLIPVisionModelTaskPool
      .evaluate(model: CLIPVisionModel | CLIPVisionTransformer) -> dict
      Returns {"<task_name>": {"accuracy": float, "loss": float}, ..., "average": {...},
               "model_info": {...}}
      Source: fusion_bench-0.2.32/fusion_bench/taskpool/clip_vision/taskpool.py

  Taskpool / eval (compat / v0.1.x):
    fusion_bench.compat.taskpool.clip_image_classification.CLIPImageClassificationTaskPool
      .evaluate(model: CLIPVisionModel) -> dict (same shape)
    Source: fusion_bench-0.2.32/fusion_bench/compat/taskpool/clip_image_classification.py

  Specialist / per-task accuracy (obtained from evaluating individual models):
    Evaluate each task model before merging by calling evaluate(task_model, modelpool).
    FusionBench does NOT expose a cached specialist table; we build it ourselves.

  Seed-setting:
    torch.manual_seed + numpy.random.seed; no FusionBench-specific seed API.

# VERIFY-IN-KAGGLE:
#   1. Confirm BaseModelPool.from_yaml(config_path) works for the project's YAML config
#      (requires _target_: fusion_bench.modelpool.CLIPVisionModelPool in the YAML).
#   2. Confirm CLIPVisionModelTaskPool is instantiated from a separate taskpool YAML
#      or constructed directly; our configs/modelpool/ YAML does NOT include taskpool config.
#   3. The FusionBench canonical 8-task taskpool YAML is at:
#      fusion_bench_config/taskpool/CLIPVisionModelTaskPool/clip-vit-classification_TA8.yaml
#      (ships with the sdist). Adapt it or use the from_modelpool helper below.
"""
from __future__ import annotations

import logging
import random

import numpy as np

log = logging.getLogger(__name__)

# Names of the 8 tasks as used in the FusionBench 8-task modelpool
# Source: fusion_bench-0.2.32/fusion_bench_config/model/clip-vit/clip-vit-base-patch16_TA8_lora.yaml
EIGHT_TASK_NAMES = [
    "sun397",
    "stanford-cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
]


def set_seed(seed: int = 2026) -> None:
    """Set numpy, random, and torch seeds for reproducibility. Pin seed=2026 per spec."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fusionbench_available() -> bool:
    try:
        import fusion_bench  # noqa: F401
        return True
    except ImportError:
        return False


def require_fusionbench():
    if not fusionbench_available():
        raise SystemExit(
            "FusionBench not installed. On Kaggle/Colab:\n"
            "  pip install git+https://github.com/tanganke/fusion_bench.git\n"
            "The OT core (src/ot_lora_merge) runs without it; FusionBench is only for eval."
        )


def load_modelpool(config_path: str):
    """Load the FusionBench modelpool from a YAML config.

    Uses the modern v0.2.x BaseModelPool.from_yaml() which dispatches via _target_.
    The project YAML (configs/modelpool/clip_vit_b32_knots_8task_lora.yaml) must contain:
      _target_: fusion_bench.modelpool.CLIPVisionModelPool

    Source: fusion_bench-0.2.32/fusion_bench/mixins/serialization.py from_yaml()
    # VERIFY-IN-KAGGLE: if the YAML does not have _target_, use the compat path:
    #   from omegaconf import OmegaConf
    #   from fusion_bench.compat.modelpool import load_modelpool_from_config
    #   cfg = OmegaConf.load(config_path)
    #   return load_modelpool_from_config(cfg)
    """
    require_fusionbench()
    from fusion_bench.modelpool import BaseModelPool  # noqa: E402
    # VERIFY-IN-KAGGLE: BaseModelPool.from_yaml dispatches to CLIPVisionModelPool via _target_
    return BaseModelPool.from_yaml(config_path)


def build_taskpool(modelpool, batch_size: int = 64, num_workers: int = 2):
    """Build a CLIPVisionModelTaskPool for the 8-task ViT-B/32 benchmark.

    Creates the taskpool directly from known config rather than a separate YAML,
    so this is self-contained. Requires FusionBench + torch.

    The canonical FusionBench 8-task test datasets are loaded via:
      fusion_bench.tasks.clip_classification.get_classnames_and_templates(task_name)
    Source: fusion_bench-0.2.32/fusion_bench/taskpool/clip_vision/taskpool.py

    # VERIFY-IN-KAGGLE: if this helper raises, instantiate CLIPVisionModelTaskPool directly
    # from the shipped YAML:
    #   from fusion_bench.taskpool.clip_vision import CLIPVisionModelTaskPool
    #   taskpool = CLIPVisionModelTaskPool.from_yaml(
    #       "path/to/fusion_bench_config/taskpool/CLIPVisionModelTaskPool/"
    #       "clip-vit-classification_TA8.yaml"
    #   )
    """
    require_fusionbench()
    from omegaconf import OmegaConf
    from fusion_bench.taskpool.clip_vision import CLIPVisionModelTaskPool  # noqa: E402

    # VERIFY-IN-KAGGLE: these dataset names are the canonical FusionBench names;
    # each resolves via fusion_bench.tasks.clip_classification.get_classnames_and_templates()
    # Source: fusion_bench-0.2.32/fusion_bench/taskpool/clip_vision/taskpool.py
    test_datasets_cfg = OmegaConf.create(
        {
            task: {
                "_target_": "fusion_bench.dataset.CLIPDataset",
                # Dataset configs are resolved internally by CLIPVisionModelTaskPool
                # via get_classnames_and_templates(task_name).
                # VERIFY-IN-KAGGLE: may need explicit dataset configs if the above fails.
            }
            for task in EIGHT_TASK_NAMES
        }
    )

    # The base CLIP model provides both the text encoder (for zero-shot classification)
    # and the image encoder placeholder that will be replaced by the merged model.
    clip_model_name = "openai/clip-vit-base-patch32"
    processor_name = clip_model_name

    taskpool_cfg = OmegaConf.create(
        {
            "_target_": "fusion_bench.taskpool.clip_vision.CLIPVisionModelTaskPool",
            "test_datasets": test_datasets_cfg,
            "processor": processor_name,
            "clip_model": clip_model_name,
            "dataloader_kwargs": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": False,
                "pin_memory": True,
            },
        }
    )

    from fusion_bench.utils import instantiate
    taskpool = instantiate(taskpool_cfg)
    return taskpool


def evaluate(merged_model, modelpool, taskpool=None, seed: int = 2026) -> dict:
    """Evaluate a merged (or specialist) model on all 8 tasks.

    Args:
        merged_model: CLIPVisionModel/CLIPVisionTransformer returned by the merge algorithm,
            OR None to get specialist accuracies (evaluates each task model individually).
        modelpool:    The loaded CLIPVisionModelPool.
        taskpool:     Optional pre-built CLIPVisionModelTaskPool; built lazily if None.
        seed:         RNG seed for reproducibility.

    Returns:
        dict: {task_name: float, "average": float}.  Flat accuracy dict (not nested).
              The "model_info" key from FusionBench's evaluate() is stripped.

    Notes on specialist evaluation (merged_model is None):
        FusionBench does NOT expose a specialist-accuracy cache. We evaluate each task
        model individually from the modelpool and return {task_name: accuracy, "average": ...}.
        Source: CLIPVisionModelTaskPool.evaluate() in
        fusion_bench-0.2.32/fusion_bench/taskpool/clip_vision/taskpool.py

    # VERIFY-IN-KAGGLE: if the flat-dict return causes KeyError in the runner, inspect
    #   the raw FusionBench report dict structure with:
    #     import json; print(json.dumps(report, indent=2))
    #   and adjust the extraction logic below to match the actual key layout.
    """
    require_fusionbench()
    set_seed(seed)

    if taskpool is None:
        taskpool = build_taskpool(modelpool)

    if merged_model is None:
        # Specialist evaluation: evaluate each task model individually
        # to obtain the per-task specialist accuracy baseline.
        accs: dict = {}
        for task_name in modelpool.model_names:
            task_model = modelpool.load_model(task_name)
            # Unwrap PeftModel to a plain vision model for evaluation
            if hasattr(task_model, "merge_and_unload"):
                task_model = task_model.merge_and_unload()
            report = taskpool.evaluate(task_model, name=task_name)
            # Extract only the relevant task's accuracy from the report
            # VERIFY-IN-KAGGLE: report structure is {"<task>": {"accuracy": float}, ...}
            # For a specialist we only care about its own task.
            if task_name in report and "accuracy" in report[task_name]:
                accs[task_name] = report[task_name]["accuracy"]
            elif task_name in report:
                # fallback: some FusionBench versions nest differently
                accs[task_name] = report[task_name]
        accs["average"] = sum(accs.values()) / len(accs) if accs else 0.0
        return accs

    # Merged model evaluation
    report = taskpool.evaluate(merged_model)
    # Flatten to {task_name: accuracy, "average": float}
    flat: dict = {}
    for k, v in report.items():
        if k == "model_info":
            continue
        if isinstance(v, dict) and "accuracy" in v:
            flat[k] = v["accuracy"]
        elif isinstance(v, (int, float)):
            flat[k] = float(v)
    if "average" not in flat:
        task_accs = [v for k, v in flat.items() if k != "average"]
        flat["average"] = sum(task_accs) / len(task_accs) if task_accs else 0.0
    return flat
