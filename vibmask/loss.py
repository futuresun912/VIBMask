"""Loss components for VIBMask.

Implements:
  - Per-feature weights ω_i = β·H(X_i) − γ·I(X_i; Y) (paper Theorem 1).
    Continuous features are discretised into 5 bins by mean ± k·std
    (Appendix A3).
  - The Gaussian-CDF gate regulariser  Σ_i ω_i Φ(μ_i / σ)  (Appendix A4).
  - A bootstrap weighting helper (Algorithm 1 line 4).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import mutual_info_score

_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Feature-weight estimation (entropy & MI through 5-bin discretisation).
# ---------------------------------------------------------------------------

def _discretise(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """5-bin discretisation by μ ± kσ (Appendix A3.1).

    Returns integer bin labels in {0, ..., n_bins-1} per column.
    Falls back to a single bin if the column is constant.
    """
    if n_bins != 5:
        # Generic equal-frequency fallback — not used in the paper.
        from sklearn.preprocessing import KBinsDiscretizer

        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        return disc.fit_transform(x.reshape(-1, 1)).astype(np.int64).ravel()

    mu = x.mean()
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    edges = np.array([mu - 2 * sd, mu - sd, mu + sd, mu + 2 * sd])
    # bin indices in {0,1,2,3,4}
    bins = np.digitize(x, edges)
    return bins.astype(np.int64)


def _entropy(labels: np.ndarray) -> float:
    """Shannon entropy of integer labels in nats."""
    _, counts = np.unique(labels, return_counts=True)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _mutual_information(x_disc: np.ndarray, y: np.ndarray) -> float:
    """Mutual information I(X_disc; Y) in nats via sklearn's contingency MI."""
    return float(mutual_info_score(x_disc, y))


def compute_feature_weights(
    X: np.ndarray,
    y: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
    n_bins: int = 5,
) -> np.ndarray:
    """Compute ω_i = β·H(X_i) − γ·I(X_i; Y) for every feature.

    Both H and I are estimated from the discretised feature columns; the
    label vector y is treated as already discrete.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D [N, d]")
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError("y must have length N")
    y_int = np.asarray(y).astype(np.int64).reshape(-1)

    weights = np.empty(d, dtype=np.float64)
    for i in range(d):
        xi = _discretise(np.asarray(X[:, i], dtype=np.float64), n_bins=n_bins)
        h_i = _entropy(xi)
        mi = _mutual_information(xi, y_int)
        weights[i] = beta * h_i - gamma * mi
    return weights


# ---------------------------------------------------------------------------
# Gate regulariser   Σ_i ω_i Φ(μ_i / σ)
# ---------------------------------------------------------------------------

def gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Element-wise Φ(x) using torch.erf."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def gate_regularizer(
    mus: torch.Tensor,
    weights: torch.Tensor,
    sigma: torch.Tensor | float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute (1/K) · Σ_k mean_n[ Σ_i ω_i Φ(μ_{k,n,i} / σ) ].

    Args:
        mus:    [K, B, d] selector logits.
        weights: [d] per-feature weights ω.
        sigma:  scalar (tensor or float) noise std σ.
        sample_weights: optional [K, B] non-negative weights used for the
            inner mean-over-samples (e.g. bootstrap multiplicities). The
            weights are normalised per-selector to sum to 1 across the batch.

    Returns:
        Scalar regulariser term to add to the prediction loss.
    """
    if mus.dim() != 3:
        raise ValueError("mus must be [K, B, d]")
    if weights.dim() != 1 or weights.shape[0] != mus.shape[-1]:
        raise ValueError("weights must be 1-D with length d matching mus")

    K = mus.shape[0]
    sigma_t = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(
        float(sigma), dtype=mus.dtype, device=mus.device
    )
    phi = gaussian_cdf(mus / sigma_t)            # [K, B, d]
    per_sample = (phi * weights).sum(dim=-1)     # [K, B]

    if sample_weights is None:
        per_selector = per_sample.mean(dim=-1)   # [K]
    else:
        if sample_weights.shape != per_sample.shape:
            raise ValueError("sample_weights must be [K, B] matching mus")
        sw = sample_weights.to(per_sample.dtype)
        denom = sw.sum(dim=-1).clamp_min(1e-12)
        per_selector = (per_sample * sw).sum(dim=-1) / denom

    return per_selector.sum() / K


# ---------------------------------------------------------------------------
# Bootstrap helpers — paper Algorithm 1, line 4 ("bootstrap dataset D_k").
# ---------------------------------------------------------------------------

def bootstrap_weights(
    batch_size: int,
    num_selectors: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-batch multinomial bootstrap weights for K selectors.

    Each row k is the count of how many times each in-batch example is drawn
    when sampling B items with replacement from the batch. This faithfully
    re-implements bagging at minibatch granularity (~63.2% unique draws per
    selector in expectation), without paying the K× memory cost of literally
    materialising K parallel mini-batches.
    """
    if batch_size <= 0 or num_selectors <= 0:
        raise ValueError("batch_size and num_selectors must be positive")
    idx = torch.randint(
        0,
        batch_size,
        (num_selectors, batch_size),
        device=device,
        generator=generator,
    )
    # one_hot is on CPU/GPU device of idx; counts via scatter_add
    weights = torch.zeros(num_selectors, batch_size, device=device, dtype=torch.float32)
    weights.scatter_add_(
        1, idx, torch.ones_like(idx, dtype=torch.float32)
    )
    return weights  # [K, B]
