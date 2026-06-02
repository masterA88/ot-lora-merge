"""OT-LoRA-Merge: merge LoRA adapters via optimal transport in the rank-r subspace.

Public API:
    extract_directions  -- ΔW = BA  ->  (U, sigma, V, p) singular-direction distribution
    merge_layer         -- merge a list of layer adapters (align | barycenter | gw)
    merge_model         -- merge a dict of {layer_name: [adapters]} across a model
    refactor            -- ΔW -> rank-R LoRA factors (A*, B*)
"""

from .directions import extract_directions, refactor
from .merge import merge_layer, merge_model

__all__ = [
    "extract_directions",
    "refactor",
    "merge_layer",
    "merge_model",
]

__version__ = "0.1.0"
