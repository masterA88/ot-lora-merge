"""Evaluation metrics. Headline = average normalized accuracy (Spec §4)."""
from __future__ import annotations

import numpy as np


def normalized_accuracy(merged_acc: float, specialist_acc: float) -> float:
    """normalized_acc = merged_acc / specialist_acc (per task)."""
    if specialist_acc <= 0:
        return 0.0
    return merged_acc / specialist_acc


def average_normalized_accuracy(merged: dict, specialist: dict) -> float:
    """Mean over tasks of merged_acc[task] / specialist_acc[task]. The single headline number."""
    tasks = list(merged.keys())
    vals = [normalized_accuracy(merged[t], specialist[t]) for t in tasks]
    return float(np.mean(vals))


def subspace_overlap(U_i, U_j) -> float:
    """Mean principal-angle cosine between two direction bases — the 'overlap' axis of the
    overlap-vs-gain diagnostic (Spec §7). 1.0 = identical subspace, 0.0 = orthogonal."""
    s = np.linalg.svd(U_i.T @ U_j, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))
