"""Smoke tests that exercise the public API end-to-end on tiny inputs.

These tests verify the package installs, imports, trains and infers — they
do not check numerical accuracy. Run with:

    pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from vibmask import (
    TrainConfig, VIBMask, accuracy, compute_feature_weights,
    feature_metrics, generate_synthetic, predict_with_masks, train_vibmask,
)


def test_generate_synthetic_shapes():
    ds = generate_synthetic("syn1", n=128, seed=0)
    assert ds.X.shape == (128, 11)
    assert ds.y.shape == (128,)
    assert ds.gt.shape == (128, 11)


def test_feature_weights_runs_on_iid_gaussian():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 11)).astype(np.float32)
    y = rng.integers(0, 2, size=200).astype(np.int64)
    w = compute_feature_weights(X, y, beta=1.0, gamma=1.0)
    assert w.shape == (11,)
    assert np.isfinite(w).all()


def test_train_one_epoch_smoke():
    ds = generate_synthetic("syn1", n=256, seed=0, alpha=5.0)
    cfg = TrainConfig(
        batch_size=32, max_iters=100, log_every=50,
        num_selectors=2, sel_hidden=(32, 32), pred_hidden=(32,),
        beta=1e-3, gamma=1e-3, seed=0,
    )
    model, weights, hist = train_vibmask(ds.X, ds.y, output_dim=2, config=cfg)
    assert isinstance(model, VIBMask)
    assert weights.shape == (11,)
    assert len(hist.iters) > 0

    # Inference path
    pred = predict_with_masks(model, ds.X, device="cpu", hard=True)
    assert pred["preds"].shape == (256,)
    assert pred["mask"].shape == (256, 11)
    # Metric path
    fm = feature_metrics(ds.gt, pred["mask"])
    assert {"tpr_mean", "fdr_mean"}.issubset(fm.keys())


def test_predict_soft_and_hard_match_shapes():
    ds = generate_synthetic("syn4", n=128, seed=0, alpha=3.0)
    cfg = TrainConfig(
        batch_size=32, max_iters=50, log_every=50,
        num_selectors=2, sel_hidden=(32,), pred_hidden=(32,),
        beta=0.0, gamma=0.0, seed=0,
    )
    model, _, _ = train_vibmask(ds.X, ds.y, output_dim=2, config=cfg)
    soft = predict_with_masks(model, ds.X, hard=False)
    hard = predict_with_masks(model, ds.X, hard=True)
    assert soft["mask"].shape == hard["mask"].shape == (128, 11)
    # Hard mask is binary, soft mask is in [0, 1]
    assert ((hard["hard_mask"] == 0) | (hard["hard_mask"] == 1)).all()
    assert ((soft["soft_mask"] >= 0) & (soft["soft_mask"] <= 1)).all()
