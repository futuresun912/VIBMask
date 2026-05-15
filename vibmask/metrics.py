"""Evaluation metrics for IWFS.

Implements TPR, FDR, CFSR (Appendix A6.4) and accuracy. All operate on
NumPy arrays so they can be used outside the training loop.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _per_sample_tpr_fdr(gt: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample TPR (%) and FDR (%) given ground-truth and selected masks.

    gt, sel: [N, d] in {0, 1}. We add a tiny epsilon to avoid division by zero
    on rows that have no ground-truth or no selected features.
    """
    eps = 1e-8
    sel = sel.astype(np.float64)
    gt = gt.astype(np.float64)

    tp = (sel * gt).sum(axis=1)
    fp = (sel * (1.0 - gt)).sum(axis=1)
    pos = gt.sum(axis=1)
    selected = sel.sum(axis=1)

    tpr = 100.0 * tp / (pos + eps)
    fdr = 100.0 * fp / (selected + eps)
    return tpr, fdr


def feature_metrics(
    gt: np.ndarray,
    sel: np.ndarray,
    control_flow_idx: int | None = None,
) -> dict[str, float]:
    """Compute mean TPR, FDR, and (optionally) CFSR.

    Args:
        gt:  [N, d] ground-truth binary mask.
        sel: [N, d] predicted binary mask (post-thresholding/voting).
        control_flow_idx: index of the control-flow feature for Syn4–6,
            or None to skip CFSR.
    """
    if gt.shape != sel.shape:
        raise ValueError("gt and sel must have identical shape")
    tpr, fdr = _per_sample_tpr_fdr(gt, sel)
    out = {
        "tpr_mean": float(tpr.mean()),
        "tpr_std": float(tpr.std()),
        "fdr_mean": float(fdr.mean()),
        "fdr_std": float(fdr.std()),
    }
    if control_flow_idx is not None:
        out["cfsr"] = float(100.0 * sel[:, control_flow_idx].mean())
    return out


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain top-1 accuracy. y_true and y_pred must be 1-D integer arrays."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float((y_true == y_pred).mean())


def summarise(metrics: Mapping[str, float]) -> str:
    """Human-readable one-liner used by the example scripts."""
    parts = []
    if "acc" in metrics:
        parts.append(f"acc={metrics['acc']:.3f}")
    if "tpr_mean" in metrics:
        parts.append(f"TPR={metrics['tpr_mean']:.1f}")
    if "fdr_mean" in metrics:
        parts.append(f"FDR={metrics['fdr_mean']:.1f}")
    if "cfsr" in metrics:
        parts.append(f"CFSR={metrics['cfsr']:.1f}")
    return " ".join(parts)
