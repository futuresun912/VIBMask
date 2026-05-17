"""Run VIBMask on digits-MNIST and reproduce the paper's Fig. 3 curve.

Trains the model once per seed using the JSON config, then evaluates
test accuracy at multiple top-k values (number of selected picks per
test instance). When the config sets `use_lasso_rerank: true`, the
inference top-k uses a Lasso-blended ranking (see V10/V16-A in the
MNIST_RESULTS analysis).

Usage
-----
    # Reproduce the strongest VIBMask on MNIST (V14-A recipe).
    python examples/mnist/demo_mnist.py --config mnist_v14a

    # Run the baseline (no extensions enabled) for comparison.
    python examples/mnist/demo_mnist.py --config mnist_baseline

    # Quick smoke check (1 seed, 1000 iters).
    python examples/mnist/demo_mnist.py --config mnist_v14a \\
        --n_seeds 1 --max_iters 1000

Each seed `s` reproduces both train and test subsamples exactly via
the seed argument. With the default config (3 seeds × full 60k MNIST),
the script takes ~15-30 minutes on a modern CPU core.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from vibmask import (
    TrainConfig,
    load_mnist,
    lasso_importance,
    predict_with_masks,
    train_vibmask,
    variance_log_prior,
)


CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        # Try with the .json suffix already in name
        path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No MNIST config {name!r} in {CONFIG_DIR}")
    with open(path) as fh:
        return json.load(fh)


def build_train_config(saved: dict, X_train: np.ndarray, y_train: np.ndarray,
                       seed: int, overrides: dict) -> TrainConfig:
    tc = dict(saved["train_config"])
    tc["sel_hidden"] = tuple(tc["sel_hidden"])
    tc["pred_hidden"] = tuple(tc["pred_hidden"])
    if "multi_k_targets" in tc:
        tc["multi_k_targets"] = tuple(tc["multi_k_targets"])
    # Build per-seed selector prior: variance log-prior.
    if tc.get("aux_soft_pred_weight", 0) > 0 or tc.get("multi_k_targets"):
        # V14-A path: variance prior baked into selector logits
        tc["selector_prior_logits"] = tuple(
            variance_log_prior(X_train).tolist()
        )
    tc.update({k: v for k, v in overrides.items() if v is not None})
    tc["seed"] = seed
    return TrainConfig(**tc)


def run_one_seed(saved: dict, seed: int, train_cfg: TrainConfig,
                 lasso_imp: np.ndarray | None, X_te: np.ndarray, y_te: np.ndarray,
                 verbose: bool) -> dict:
    md = load_mnist("mnist", n_train=saved["n_train"], n_test=saved["n_test"],
                    seed=seed)
    X_tr, y_tr = md.X_train, md.y_train

    progress_cb = None
    if verbose:
        def cb(it, info):
            if it % (5 * train_cfg.log_every) == 0:
                print(f"  [{it:5d}] train={info['train_loss']:.4f} "
                      f"(pred={info['train_pred']:.4f} reg={info['train_reg']:.4f}) "
                      f"val={info['val_loss']:.4f} val_acc={info['val_acc']:.3f}")
        progress_cb = cb

    t0 = time.time()
    model, _, _ = train_vibmask(
        X_tr, y_tr, output_dim=10,
        config=replace(train_cfg, seed=seed),
        progress_cb=progress_cb,
    )
    train_time = time.time() - t0

    eval_ks = saved["eval_ks"]
    use_lasso = bool(saved.get("use_lasso_rerank", False))
    eval_kernel = int(saved.get("eval_patch_kernel", 3))
    prior_blend = float(saved.get("prior_blend", 0.0))
    img_hw = 28

    rows = []
    for k in eval_ks:
        nms = 2 if k <= 10 else 0
        pred = predict_with_masks(
            model, md.X_test, device=train_cfg.device, hard=True,
            top_k=int(k), select_patch=True, img_hw=img_hw,
            patch_kernel=eval_kernel, nms_radius=nms,
            prior_score=(lasso_imp if use_lasso else None),
            prior_blend=prior_blend if use_lasso else 0.0,
        )
        rows.append({
            "k": int(k),
            "n_sel_per_instance": float(pred["hard_mask"].sum(axis=1).mean()),
            "acc": float((pred["preds"] == md.y_test).mean()),
        })
    return {"seed": seed, "train_time_s": train_time, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="mnist_v14a",
                   help="Config name in examples/mnist/configs/ "
                        "(without .json). Default: mnist_v14a.")
    p.add_argument("--n_seeds", type=int, default=None,
                   help="Override n_seeds from the config.")
    p.add_argument("--n_train", type=int, default=None,
                   help="Override n_train (≤ 60000).")
    p.add_argument("--n_test", type=int, default=None,
                   help="Override n_test (≤ 10000).")
    p.add_argument("--max_iters", type=int, default=None,
                   help="Override training max_iters (for quick smoke).")
    p.add_argument("--device", default=None,
                   help="cpu or cuda; default from config (cpu).")
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    saved = load_config(args.config)
    if args.n_seeds is not None: saved["n_seeds"] = args.n_seeds
    if args.n_train is not None: saved["n_train"] = args.n_train
    if args.n_test is not None:  saved["n_test"] = args.n_test
    overrides = {}
    if args.max_iters is not None: overrides["max_iters"] = args.max_iters
    if args.device is not None:    overrides["device"] = args.device

    # Pre-load training data once (for Lasso fit + prior construction).
    print(f"=== VIBMask MNIST demo: config={args.config} ===")
    print(f"  n_train={saved['n_train']}, n_test={saved['n_test']}, "
          f"n_seeds={saved['n_seeds']}, eval_ks={saved['eval_ks']}")
    md0 = load_mnist("mnist", n_train=saved["n_train"], n_test=saved["n_test"],
                     seed=args.seed_start)
    lasso_imp = None
    if saved.get("use_lasso_rerank", False):
        print(f"  Fitting Lasso (one-vs-rest L1-logistic) for inference prior...")
        t = time.time()
        lasso_imp = lasso_importance(md0.X_train, md0.y_train, C=0.1,
                                     seed=args.seed_start)
        print(f"  Lasso fit in {time.time()-t:.1f}s "
              f"(nonzero={int((lasso_imp>0.01).sum())}/784).")

    per_seed = []
    for s in range(args.seed_start, args.seed_start + saved["n_seeds"]):
        print(f"\n--- seed {s} ---")
        train_cfg = build_train_config(saved, md0.X_train, md0.y_train,
                                       seed=s, overrides=overrides)
        r = run_one_seed(saved, s, train_cfg, lasso_imp, md0.X_test, md0.y_test,
                         args.verbose)
        print(f"  train_time={r['train_time_s']:.1f}s")
        for row in r["rows"]:
            print(f"   k={row['k']:>3} picks → n_active={row['n_sel_per_instance']:6.1f} "
                  f"pixels  acc={row['acc']:.4f}")
        per_seed.append(r)

    # Aggregate across seeds.
    print(f"\n=== MNIST summary over {len(per_seed)} seeds ===")
    eval_ks = saved["eval_ks"]
    print(f"  {'k':>5}  {'mean active px':>15}  {'mean acc':>12}  {'std acc':>9}")
    for k in eval_ks:
        accs = np.array([row["acc"] for r in per_seed for row in r["rows"]
                         if row["k"] == k])
        nsels = np.array([row["n_sel_per_instance"] for r in per_seed
                          for row in r["rows"] if row["k"] == k])
        print(f"  {k:>5}  {nsels.mean():>15.1f}  {accs.mean():.4f}     "
              f"{accs.std():.4f}")


if __name__ == "__main__":
    main()
