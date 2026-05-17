"""MNIST data loader + feature-importance priors for the MNIST extension.

This module is **independent** of the synthetic pipeline — the original
`data.py` is unchanged. Loaders here support the MNIST experiment in
`examples/mnist/`.

The Lasso / variance helpers compute per-feature priors that can be fed
into:

  - `TrainConfig.selector_prior_logits` — training-time bias (warm-start
    or hard-prune).
  - `TrainConfig.feature_weight_offset` — additive regulariser term.
  - `predict_with_masks(prior_score=...)` — inference-time top-k rerank.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


_DEFAULT_CACHE = os.path.expanduser("~/.cache/vibmask_mnist")


@dataclass
class MnistDataset:
    name: str
    X_train: np.ndarray  # [N_train, 784] float32 in [0, 1]
    y_train: np.ndarray  # [N_train]    int64 in {0..9}
    X_test:  np.ndarray  # [N_test,  784]
    y_test:  np.ndarray  # [N_test]


def _fetch_openml_mnist(name: str, cache_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch digits-MNIST via sklearn's fetch_openml (60k+10k = 70k total)."""
    from sklearn.datasets import fetch_openml
    os.makedirs(cache_dir, exist_ok=True)
    old_env = os.environ.get("SCIKIT_LEARN_DATA")
    os.environ["SCIKIT_LEARN_DATA"] = cache_dir
    try:
        data_id = {"mnist": 554}[name]
        ds = fetch_openml(data_id=data_id, as_frame=False)
    finally:
        if old_env is None:
            del os.environ["SCIKIT_LEARN_DATA"]
        else:
            os.environ["SCIKIT_LEARN_DATA"] = old_env
    X = np.asarray(ds.data, dtype=np.float32) / 255.0  # [N, 784] in [0, 1]
    y_raw = np.asarray(ds.target).reshape(-1)
    y = np.asarray([int(v) for v in y_raw], dtype=np.int64)
    return X, y


def load_mnist(
    name: str = "mnist",
    n_train: int = 60_000,
    n_test: int = 10_000,
    seed: int = 0,
    cache_dir: str = _DEFAULT_CACHE,
) -> MnistDataset:
    """Load digits-MNIST. Default = full benchmark split (60k / 10k).

    Args:
        name:    only 'mnist' (digits-MNIST) is supported in this release.
        n_train: cap on training samples (stratified subsample).
        n_test:  cap on test samples (stratified subsample).
        seed:    subsample seed.

    Returns: MnistDataset with X in [0, 1] float32 and y in {0..9} int64.
    """
    name = name.lower()
    if name != "mnist":
        raise ValueError(f"only 'mnist' is supported (got {name!r})")
    X, y = _fetch_openml_mnist(name, cache_dir)
    X_tr_full, y_tr_full = X[:60_000], y[:60_000]
    X_te_full, y_te_full = X[60_000:], y[60_000:]
    from sklearn.model_selection import train_test_split
    if n_train < 60_000:
        tr_idx, _ = train_test_split(
            np.arange(60_000), train_size=n_train,
            stratify=y_tr_full, random_state=seed,
        )
        X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
    else:
        X_tr, y_tr = X_tr_full, y_tr_full
    if n_test < 10_000:
        te_idx, _ = train_test_split(
            np.arange(10_000), train_size=n_test,
            stratify=y_te_full, random_state=seed + 1,
        )
        X_te, y_te = X_te_full[te_idx], y_te_full[te_idx]
    else:
        X_te, y_te = X_te_full, y_te_full
    return MnistDataset(name=name, X_train=X_tr, y_train=y_tr,
                        X_test=X_te, y_test=y_te)


def lasso_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    seed: int = 0,
    max_iter: int = 200,
    subsample: int = 10_000,
) -> np.ndarray:
    """Per-feature importance from one-vs-rest L1-logistic regression.

    Returns per-feature max-abs coefficient across classes, normalised
    to [0, 1]. Used as a prior on the selector — features with high
    Lasso importance are *a priori* informative.

    Args:
        C: inverse regularisation strength (smaller = sparser). 0.1
           gives a useful but not over-sparse Lasso solution on MNIST.
        subsample: cap on training rows; Lasso doesn't need all 60k
           samples for stable per-feature scores.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    n = X_train.shape[0]
    if n > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=subsample, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    base = LogisticRegression(
        penalty="l1", solver="liblinear", C=C,
        max_iter=max_iter, random_state=seed,
    )
    clf = OneVsRestClassifier(base, n_jobs=4)
    clf.fit(X_train, y_train)
    coef = np.abs(np.stack(
        [est.coef_.ravel() for est in clf.estimators_], axis=0
    ))
    score = coef.max(axis=0) if coef.ndim == 2 else coef
    m = score.max()
    if m > 1e-12:
        score = score / m
    return score.astype(np.float32)


def variance_log_prior(X_train: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-feature log-variance prior (zero-mean centered).

    For `TrainConfig.selector_prior_logits`: pushes the selector away
    from features whose H ≈ 0 (always-zero pixels on MNIST). Returns
    a float32 array of shape (d,) centered at zero.
    """
    var = X_train.var(axis=0).astype(np.float32)
    log_v = np.log(var + eps)
    return (log_v - log_v.mean()).astype(np.float32)


def hard_prune_prior(
    base_prior: np.ndarray,
    importance: np.ndarray,
    bottom_frac: float = 0.5,
    block_logit: float = -1000.0,
) -> np.ndarray:
    """Combine a base prior with a hard-prune mask from importance scores.

    For `TrainConfig.selector_prior_logits` (V16-A recipe): set the
    `bottom_frac` of features with the lowest `importance` to
    `block_logit` (≈ -1000) so the selector cannot pick them. The
    remaining positions keep the `base_prior` value.

    Args:
        base_prior:  [d] e.g. from `variance_log_prior`.
        importance: [d] e.g. from `lasso_importance`.
        bottom_frac: fraction (0..1) of features to block.
        block_logit: very-negative bias to pin those gates to ≈0.
    """
    if base_prior.shape != importance.shape:
        raise ValueError("base_prior and importance must have the same shape")
    sorted_imp = np.sort(importance)
    threshold = sorted_imp[int(len(sorted_imp) * float(bottom_frac))]
    out = base_prior.astype(np.float32).copy()
    out[importance <= threshold] = float(block_logit)
    return out
