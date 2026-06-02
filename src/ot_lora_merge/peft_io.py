"""Load/save PEFT LoRA adapters and convert to the numpy {A,B} convention used by the merger.

GPU/torch + PEFT are only needed here and in the eval harness — the OT core stays pure-numpy.
This module is a thin, well-defined boundary.

FusionBench API surface (sourced from fusion_bench-0.2.32 sdist,
file fusion_bench/models/linearized/vision_model.py):
  - load_lora_vision_model_hf(base_model_name, peft_name, merge_and_unload=False)
    returns a PeftModel wrapping CLIPVisionTransformer with lora layers NOT yet merged.
  - The PEFT lora layer keys follow the pattern:
      base_model.model.<submodule>.lora_A.default.weight   shape (r, in)
      base_model.model.<submodule>.lora_B.default.weight   shape (out, r)
    (PEFT >= 0.11 appended the adapter name "default" between lora_A/lora_B and weight.)
  - alpha/r scaling is stored in peft_model.peft_config['default'].lora_alpha and .r

PEFT key convention (verified against peft >= 0.11 source and FusionBench TA8_lora configs):
  state_dict key pattern:
    base_model.model.<module_path>.lora_A.default.weight  -> A matrix (r x n)
    base_model.model.<module_path>.lora_B.default.weight  -> B matrix (m x r)
  Effective delta: ΔW = (alpha / r) * B @ A
  alpha and r are read from peft_model.peft_config[adapter_name].

Key normalisation applied here:
  Strip "base_model.model." prefix, then strip ".lora_A.default" / ".lora_B.default"
  to recover the plain submodule path used as layer_name.

  # CONFIRMED-IN-KAGGLE 2026-06-01 (scripts/kaggle_probe.py): the installed PEFT uses the "default"
  # adapter-name segment. PeftModel state_dict keys are exactly
  #   base_model.model.vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.lora_A.default.weight
  # The raw adapter weights use the no-"default", no-"vision_model" form
  #   base_model.model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.lora_A.weight
  # _strip_key() handles both. This module is convention-agnostic about WHICH modules carry LoRA.
  #
  # The v1 benchmark adapters are KnOTS B/32 (hoffman-lab/KnOTS-ViT-B-32_lora_R16_*):
  #   r=16, alpha=16 -> scale 1.0, target_modules=[q,k,v,out]_proj, A=(16,768) B=(768,16),
  #   base_model_name_or_path=null (pass openai/clip-vit-base-patch32), weights = adapter_model.bin.
  # (probe2 also verified the mechanism on tanganke B/16 adapters: scale 2.0, q/v only.)
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
#  Internal key-normalisation helpers                                          #
# --------------------------------------------------------------------------- #

_LORA_A_SUFFIXES = (
    ".lora_A.default.weight",   # PEFT >= 0.11 (adapter named "default")
    ".lora_A.weight",           # PEFT < 0.11 fallback
)
_LORA_B_SUFFIXES = (
    ".lora_B.default.weight",
    ".lora_B.weight",
)
_BASE_PREFIX = "base_model.model."


def _strip_key(key: str, suffixes: tuple) -> str | None:
    """Return the bare layer name if key ends with one of the given suffixes, else None."""
    k = key
    if k.startswith(_BASE_PREFIX):
        k = k[len(_BASE_PREFIX):]
    for suf in suffixes:
        if k.endswith(suf):
            return k[: -len(suf)]
    return None


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def lora_state_to_layers(state_dict, scale: float | None = None) -> dict:
    """Convert a PEFT LoRA state_dict into {layer_name: {"A","B","scale"}} (numpy).

    Handles both PEFT >= 0.11 ("default" adapter name in key) and < 0.11 (no adapter name).

    PEFT convention (>= 0.11):
        base_model.model.<module>.lora_A.default.weight   shape (r, n)  — the A matrix
        base_model.model.<module>.lora_B.default.weight   shape (m, r)  — the B matrix
    Effective update: ΔW = (alpha/r) · B @ A.  Pass `scale` to override alpha/r.

    Source: peft >= 0.11 source + fusion_bench-0.2.32/fusion_bench/models/linearized/vision_model.py
    """
    import numpy as np  # local import: keep core import-light

    layers: dict = {}
    a_map: dict = {}
    b_map: dict = {}

    for key, w in state_dict.items():
        layer = _strip_key(key, _LORA_A_SUFFIXES)
        if layer is not None:
            a_map[layer] = w
            continue
        layer = _strip_key(key, _LORA_B_SUFFIXES)
        if layer is not None:
            b_map[layer] = w

    for layer_name in a_map:
        if layer_name in b_map:
            A = np.asarray(a_map[layer_name].detach().cpu().float().numpy())
            B = np.asarray(b_map[layer_name].detach().cpu().float().numpy())
            layers[layer_name] = {"A": A, "B": B, "scale": scale}

    return layers


def get_lora_scale(peft_model, adapter_name: str = "default") -> float:
    """Read alpha/r from a PeftModel's config and return the effective LoRA scale.

    Source: peft.config.LoraConfig attributes .lora_alpha and .r
    # CONFIRMED-IN-KAGGLE 2026-06-01: peft_config.keys() == ['default']; alpha=32, r=16 -> 2.0.
    """
    cfg = peft_model.peft_config[adapter_name]
    return float(cfg.lora_alpha) / float(cfg.r)


def peft_model_to_layers(peft_model, adapter_name: str = "default") -> dict:
    """Convenience wrapper: read scale from peft_model and call lora_state_to_layers.

    peft_model should be a PeftModel (merge_and_unload=False) wrapping a CLIPVisionTransformer.
    Returns {layer_name: {"A","B","scale"}} ready for merge_model().

    Source: fusion_bench-0.2.32/fusion_bench/models/linearized/vision_model.py —
    load_lora_vision_model_hf with merge_and_unload=False produces exactly this shape.
    # CONFIRMED-IN-KAGGLE 2026-06-01: the "base_model.model." prefix + ".lora_A.default.weight"
    # pattern holds for tanganke/clip-vit-base-patch16_*_lora-16 (48 lora tensors / adapter).
    """
    scale = get_lora_scale(peft_model, adapter_name)
    state_dict = peft_model.state_dict()
    return lora_state_to_layers(state_dict, scale=scale)


def layers_to_lora_state(merged_layers: dict, reference_state: dict) -> dict:
    """Write merged (A*, B*) factors back into a PEFT-shaped state_dict.

    Clones dtype and device from `reference_state` (a state_dict of the original PeftModel)
    for each key so the result can be loaded back with peft_model.load_state_dict().

    Strategy:
      1. For every layer_name in merged_layers, reconstruct the PEFT key(s) by trying
         both the >= 0.11 and < 0.11 suffix forms against keys actually present in
         reference_state.
      2. Write torch tensors with the correct dtype/device cloned from reference_state.
      3. Return a *copy* of reference_state with the lora_A / lora_B values replaced.
         Non-LoRA keys (embedding, norm, etc.) are left unchanged.

    Source: PEFT >= 0.11 state_dict key convention (see module docstring).
    # VERIFY-IN-KAGGLE: after layers_to_lora_state, call
    #   peft_model.load_state_dict(new_state, strict=False)
    # and verify missing_keys is empty for lora_A/lora_B entries.
    """
    import copy

    import numpy as np
    import torch

    new_state = copy.copy(reference_state)  # shallow copy; values replaced below

    # Build a lookup: bare_layer_name -> (a_key, b_key) as they appear in reference_state
    a_keys: dict = {}
    b_keys: dict = {}
    for key in reference_state:
        layer = _strip_key(key, _LORA_A_SUFFIXES)
        if layer is not None:
            a_keys[layer] = key
            continue
        layer = _strip_key(key, _LORA_B_SUFFIXES)
        if layer is not None:
            b_keys[layer] = key

    for layer_name, (A_np, B_np) in merged_layers.items():
        if layer_name not in a_keys or layer_name not in b_keys:
            # Layer not present in reference; skip (can happen if only a subset was merged)
            continue
        a_key = a_keys[layer_name]
        b_key = b_keys[layer_name]
        ref_a = reference_state[a_key]
        ref_b = reference_state[b_key]
        new_state[a_key] = torch.tensor(
            A_np, dtype=ref_a.dtype, device=ref_a.device
        )
        new_state[b_key] = torch.tensor(
            B_np, dtype=ref_b.dtype, device=ref_b.device
        )

    return new_state
