"""Register OT-LoRA-Merge as a FusionBench algorithm so baselines + eval are shared
(apples-to-apples). Inheriting from BaseAlgorithm gives us the from_yaml() classmethod and
satisfies the run(modelpool) contract. The class is fully import-light: FusionBench/torch
imports are deferred inside run(), so the pure-numpy OT core tests still pass without them.

FusionBench API used here (sourced from fusion_bench-0.2.32 sdist):

  BaseAlgorithm
    module : fusion_bench.method.base_algorithm
    run(self, modelpool: BaseModelPool) -> nn.Module   [abstract; we implement it]
    Source: fusion_bench-0.2.32/fusion_bench/method/base_algorithm.py

  BaseModelPool
    module : fusion_bench.modelpool.base_pool
    .model_names : List[str]   — task names (excludes "_pretrained_")
    .load_model(name) -> nn.Module  — loads a CLIPVisionTransformer (or PeftModel)
    .load_pretrained_model() -> nn.Module   — loads the base vision model
    Source: fusion_bench-0.2.32/fusion_bench/modelpool/base_pool.py

  CLIPVisionModelPool
    module : fusion_bench.modelpool.CLIPVisionModelPool
    Subclass of BaseModelPool; load_model() returns CLIPVisionModel/CLIPVisionTransformer.
    With LoRA (merge_and_unload=False) the returned object is a PeftModel.
    Source: fusion_bench-0.2.32/fusion_bench/modelpool/clip_vision/modelpool.py

  The modelpool YAML must contain _target_: fusion_bench.modelpool.CLIPVisionModelPool
  and list models with _target_: fusion_bench.models.linearized.vision_model.load_lora_vision_model_hf
  (with merge_and_unload: false) so that run() receives unmerged PeftModels.
  Source: fusion_bench-0.2.32/fusion_bench_config/modelpool/CLIPVisionModelPool/clip-vit-base-patch16_TA8_lora.yaml

  # VERIFY-IN-KAGGLE: confirm that modelpool.load_model(name) for a LoRA entry
  #   with merge_and_unload=false returns a PeftModel, not a plain CLIPVisionModel.
  #   If it returns a merged CLIPVisionModel (merge_and_unload=true), switch to
  #   lora_state_to_layers(model.state_dict()) instead — but scale won't be recoverable;
  #   use scale=1.0 and note the approximation.
"""
from __future__ import annotations

import copy
import logging

log = logging.getLogger(__name__)

from .merge import merge_model
from .peft_io import layers_to_lora_state, peft_model_to_layers


class OTLoRAMergeAlgorithm:
    """FusionBench-style merge algorithm following the BaseAlgorithm contract.

    Inherits from fusion_bench.method.base_algorithm.BaseAlgorithm at runtime
    (deferred import so the pure-numpy tests work without FusionBench installed).

    Config keys (all optional; defaults shown):
        mode        : "align" | "barycenter" | "gw"  (default "barycenter")
        cost        : "paired" | "left" | "right" | "grassmann"  (default "paired")
        eps         : float Sinkhorn regularization  (default 0.01)
        exact       : bool  use exact EMD instead of Sinkhorn  (default False)
        target_rank : int | None  output LoRA rank  (default = max input rank)
        gw_eps      : float GW regularization  (default 0.05)
    """

    def __init__(
        self,
        mode: str = "barycenter",
        cost: str = "paired",
        eps: float = 1e-2,
        exact: bool = False,
        target_rank: int | None = None,
        gw_eps: float = 5e-2,
    ):
        self.mode = mode
        self.cost = cost
        self.eps = eps
        self.exact = exact
        self.target_rank = target_rank
        self.gw_eps = gw_eps

    # ------------------------------------------------------------------ #
    #  Core implementation                                                 #
    # ------------------------------------------------------------------ #

    def run(self, modelpool):
        """Merge all task LoRA adapters in modelpool and return a merged CLIPVisionModel.

        Steps
        -----
        1. Load each task model (PeftModel with unmerged LoRA layers) from the modelpool.
        2. Extract {layer_name: {"A","B","scale"}} dicts via peft_io.peft_model_to_layers().
        3. Group into {layer_name: [adapter_t, ...]} across tasks.
        4. Call merge_model() (pure-numpy OT core).
        5. Write the merged (A*, B*) tensors back into the first task model's state_dict.
        6. Load the updated state_dict into a fresh copy of the pretrained base model.

        FusionBench contract: returns an nn.Module that the taskpool can evaluate directly.

        # VERIFY-IN-KAGGLE: if modelpool.has_pretrained is False, replace
        #   modelpool.load_pretrained_model() with modelpool.load_model(modelpool.model_names[0])
        #   and note that the base model comes from the first task model's merge_and_unload.
        """
        # Deferred imports — FusionBench/torch only available on Kaggle/Colab
        import torch

        task_names = list(modelpool.model_names)
        if len(task_names) == 0:
            raise ValueError("Modelpool has no task models.")

        log.info(
            "OT-LoRA-Merge: loading %d task adapters (mode=%s cost=%s eps=%s)",
            len(task_names),
            self.mode,
            self.cost,
            self.eps,
        )

        # 1. Load task models and extract layer dicts
        task_layers: list[dict] = []
        first_peft_model = None
        first_state_dict = None

        for name in task_names:
            model = modelpool.load_model(name)
            layers = peft_model_to_layers(model)
            task_layers.append(layers)
            if first_peft_model is None:
                first_peft_model = model
                first_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        # 2. Group across tasks: {layer_name: [{"A","B","scale"}, ...]}
        all_layer_names = list(task_layers[0].keys())
        grouped: dict = {
            ln: [tl[ln] for tl in task_layers if ln in tl]
            for ln in all_layer_names
        }
        # Drop layers where not all tasks contributed (e.g. partially-adapted models)
        grouped = {ln: ads for ln, ads in grouped.items() if len(ads) == len(task_names)}

        log.info(
            "OT-LoRA-Merge: merging %d LoRA layers across %d tasks.",
            len(grouped),
            len(task_names),
        )

        # 3. OT merge (pure-numpy, CPU)
        merged: dict = merge_model(
            grouped,
            mode=self.mode,
            cost=self.cost,
            eps=self.eps,
            exact=self.exact,
            target_rank=self.target_rank,
            gw_eps=self.gw_eps,
        )
        # merged = {layer_name: (A_star, B_star)}

        # 4. Write merged factors back and load into a fresh copy of the pretrained model
        # Load the pretrained base model (CLIPVisionModel / CLIPVisionTransformer)
        # VERIFY-IN-KAGGLE: modelpool.has_pretrained should be True when the YAML lists
        # _pretrained_: openai/clip-vit-base-patch32
        if modelpool.has_pretrained:
            base_model = modelpool.load_pretrained_model()
        else:
            # Fallback: use the first task model with LoRA merged and unloaded
            log.warning(
                "No pretrained model in pool; using first task model as base. "
                "Results may be slightly off — ensure the YAML has _pretrained_."
            )
            # VERIFY-IN-KAGGLE: this path is the fallback and should not normally be hit
            # for the standard TA8-lora modelpool config.
            base_model = copy.deepcopy(first_peft_model)
            if hasattr(base_model, "merge_and_unload"):
                base_model = base_model.merge_and_unload()

        # Write merged (A*, B*) into the first task's peft state_dict, then load
        new_state = layers_to_lora_state(merged, first_state_dict)

        # Load back into first_peft_model (in-place) to get an updated PeftModel
        first_peft_model.load_state_dict(new_state, strict=False)

        # Merge LoRA weights into the base model and return a plain CLIPVisionTransformer
        # VERIFY-IN-KAGGLE: first_peft_model.merge_and_unload() merges lora_A/lora_B into
        # the base weight and strips the PEFT adapter layers, returning a plain nn.Module
        # identical in shape to the pretrained base — exactly what CLIPVisionModelTaskPool
        # expects as input to its evaluate() method.
        merged_model = first_peft_model.merge_and_unload()
        return merged_model


def register():
    """Register OTLoRAMergeAlgorithm as a FusionBench BaseAlgorithm subclass.

    FusionBench's modern (v0.2.x) pattern: algorithms are loaded via Hydra/OmegaConf
    instantiation using the _target_ key in the YAML config, not a name registry.
    See fusion_bench-0.2.32/fusion_bench/programs/fusion_program.py _instantiate_and_setup().

    This function makes OTLoRAMergeAlgorithm a proper BaseAlgorithm subclass so it can
    be used with _target_: ot_lora_merge.fusionbench_hook.OTLoRAMergeAlgorithm in a
    method YAML config.

    Source: fusion_bench-0.2.32/fusion_bench/method/base_algorithm.py
    # VERIFY-IN-KAGGLE: after calling register(), try:
    #   from fusion_bench.method import BaseAlgorithm
    #   assert issubclass(OTLoRAMergeAlgorithm, BaseAlgorithm)
    """
    try:
        from fusion_bench.method.base_algorithm import BaseAlgorithm  # noqa: F401

        if BaseAlgorithm not in OTLoRAMergeAlgorithm.__bases__:
            # Dynamically patch the base class so FusionBench's type checks pass.
            # This is a seam: OTLoRAMergeAlgorithm already implements run() fully;
            # adding BaseAlgorithm as a base just satisfies isinstance checks.
            OTLoRAMergeAlgorithm.__bases__ = (BaseAlgorithm,)
            log.info(
                "OTLoRAMergeAlgorithm registered as BaseAlgorithm subclass."
            )
    except ImportError:
        # FusionBench not installed (e.g. running OT core tests locally) — no-op.
        pass
    return OTLoRAMergeAlgorithm
