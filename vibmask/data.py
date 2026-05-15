"""Synthetic Syn1–Syn6 datasets used in INVASE/L2X/ProtoGate/VIBMask.

For each X ~ N(0, I_d) we sample y ~ Bernoulli(σ(logit(X))). The conditional
datasets (Syn4–6) gate logit on sign(X_11), making feature 11 a
*control-flow* feature whose selection rate is reported as CFSR.

The exact logit functions follow the supplementary `data_generate_syn.py`
that ships with the paper (which differs from the abbreviated form printed
in the paper body for syn3 — we follow the supplementary which is what the
authors actually ran).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _log_logit_syn1(X: np.ndarray) -> np.ndarray:
    """Pre-sigmoid argument for Syn1: X1 · X2."""
    return X[:, 0] * X[:, 1]


def _log_logit_syn2(X: np.ndarray) -> np.ndarray:
    """Pre-sigmoid argument for Syn2: ΣᵢXᵢ² − 4 over i ∈ 3..6."""
    return np.sum(X[:, 2:6] ** 2, axis=1) - 4.0


def _log_logit_syn3(X: np.ndarray) -> np.ndarray:
    """Pre-sigmoid argument for Syn3 (supplementary form):
    −10·sin(0.2·X₇) + |X₈| + X₉ + exp(−X₁₀) − 2.4
    """
    with np.errstate(over="ignore"):
        return (
            -10.0 * np.sin(0.2 * X[:, 6])
            + np.abs(X[:, 7])
            + X[:, 8]
            + np.clip(np.exp(-X[:, 9]), 0.0, 1e6)
            - 2.4
        )


_BASE_LOG_LOGITS = {
    "syn1": _log_logit_syn1,
    "syn2": _log_logit_syn2,
    "syn3": _log_logit_syn3,
}


def _log_logit(X: np.ndarray, name: str) -> np.ndarray:
    name = name.lower()
    if name in _BASE_LOG_LOGITS:
        return _BASE_LOG_LOGITS[name](X)
    if name in ("syn4", "syn5", "syn6"):
        a, b = {
            "syn4": ("syn1", "syn2"),
            "syn5": ("syn1", "syn3"),
            "syn6": ("syn2", "syn3"),
        }[name]
        la, lb = _BASE_LOG_LOGITS[a](X), _BASE_LOG_LOGITS[b](X)
        return np.where(X[:, 10] < 0, la, lb)
    raise ValueError(f"unknown synthetic dataset {name!r}")


def _ground_truth_mask(X: np.ndarray, name: str) -> np.ndarray:
    n, d = X.shape
    gt = np.zeros((n, d), dtype=np.float32)
    name = name.lower()
    if name == "syn1":
        gt[:, :2] = 1
    elif name == "syn2":
        gt[:, 2:6] = 1
    elif name == "syn3":
        gt[:, 6:10] = 1
    elif name in ("syn4", "syn5", "syn6"):
        idx_neg = np.where(X[:, 10] < 0)[0]
        idx_pos = np.where(X[:, 10] >= 0)[0]
        gt[:, 10] = 1
        if name == "syn4":
            gt[idx_neg, :2] = 1
            gt[idx_pos, 2:6] = 1
        elif name == "syn5":
            gt[idx_neg, :2] = 1
            gt[idx_pos, 6:10] = 1
        else:  # syn6
            gt[idx_neg, 2:6] = 1
            gt[idx_pos, 6:10] = 1
    else:
        raise ValueError(f"unknown synthetic dataset {name!r}")
    return gt


@dataclass
class SynDataset:
    X: np.ndarray   # [N, d] float32
    y: np.ndarray   # [N]    int64 in {0, 1}
    gt: np.ndarray  # [N, d] float32 ground-truth feature mask
    name: str


def generate_synthetic(
    name: str,
    n: int = 5000,
    d: int = 11,
    seed: int = 0,
    alpha: float = 1.0,
) -> SynDataset:
    """Generate one of Syn1–Syn6 (paper §5 / Appendix A6.1.1).

    `alpha` scales the pre-sigmoid argument: P(Y=1) = σ(α · arg) where arg
    is the supplementary formula's pre-exp expression. α=1.0 reproduces the
    INVASE/L2X/supplementary baseline (Bayes-opt ≈ 0.63 on Syn1, 0.82 on
    Syn2/3). The paper's reported accuracies require α ≈ 2–5; we expose the
    knob so we can sweep and confirm.
    """
    if d < 11:
        raise ValueError("d must be >= 11 for the conditional datasets")
    name = name.lower()
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, d)).astype(np.float32)
    arg = _log_logit(X, name)  # pre-sigmoid argument
    # P(Y=1) = σ(α · arg). α=1 reproduces the supplementary baseline.
    arg_scaled = float(alpha) * arg
    # numerically stable σ via clipping to safe range for the binomial
    arg_scaled = np.clip(arg_scaled, -500.0, 500.0)
    p1 = 1.0 / (1.0 + np.exp(-arg_scaled))
    p1 = np.clip(p1, 1e-7, 1 - 1e-7)
    y = rng.binomial(1, p1).astype(np.int64)
    gt = _ground_truth_mask(X, name)
    return SynDataset(X=X, y=y, gt=gt, name=name)
