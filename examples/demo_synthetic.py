"""Run the strongest VIBMask configuration on one of the six synthetic
benchmarks (Syn1-Syn6) from the paper. Reports TPR / FDR / CFSR / ACC
averaged over the requested number of independent seeds.

Usage
-----
    # Reproduce the published Syn1 result with the released best config:
    python examples/demo_synthetic.py --dataset syn1

    # Run a different dataset with a custom number of seeds:
    python examples/demo_synthetic.py --dataset syn5 --n_seeds 5

    # Override individual hyperparameters from the saved config:
    python examples/demo_synthetic.py --dataset syn3 --max_iters 5000

Reproduces the per-dataset numbers reported in the paper's Table 1 and
appendix A6.4 (within seed-to-seed variance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from vibmask import (
    TrainConfig,
    feature_metrics,
    generate_synthetic,
    predict_with_masks,
    summarise,
    train_vibmask,
)


CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(dataset: str) -> dict:
    cfg_path = CONFIG_DIR / f"{dataset}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config for {dataset!r} at {cfg_path}")
    with open(cfg_path) as fh:
        return json.load(fh)


def build_train_config(saved: dict, overrides: dict) -> TrainConfig:
    tc = dict(saved["train_config"])
    tc["sel_hidden"] = tuple(tc["sel_hidden"])
    tc["pred_hidden"] = tuple(tc["pred_hidden"])
    tc.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**tc)


def run_one_seed(dataset: str, seed: int, train_cfg: TrainConfig,
                 data_alpha: float, n_train: int, n_test: int,
                 soft_inference: bool, verbose: bool):
    train_ds = generate_synthetic(dataset, n=n_train, seed=seed, alpha=data_alpha)
    test_ds  = generate_synthetic(dataset, n=n_test,  seed=seed + 100,
                                  alpha=data_alpha)

    t0 = time.time()
    progress_cb = None
    if verbose:
        def cb(it, info):
            if it % (5 * train_cfg.log_every) == 0:
                print(f"  [{it:5d}] train={info['train_loss']:.4f} "
                      f"(pred={info['train_pred']:.4f} reg={info['train_reg']:.4f}) "
                      f"val={info['val_loss']:.4f} val_acc={info['val_acc']:.3f}")
        progress_cb = cb

    model, _, _ = train_vibmask(
        train_ds.X, train_ds.y, output_dim=2,
        config=replace(train_cfg, seed=seed),
        progress_cb=progress_cb,
    )
    train_time = time.time() - t0

    pred = predict_with_masks(model, test_ds.X,
                              device=train_cfg.device,
                              hard=not soft_inference)
    cfi = 10 if dataset.lower() in {"syn4", "syn5", "syn6"} else None
    fm = feature_metrics(test_ds.gt, pred["mask"], control_flow_idx=cfi)
    fm["acc"] = float((pred["preds"] == test_ds.y).mean())
    fm["train_time_s"] = train_time
    return fm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=[f"syn{i}" for i in range(1, 7)],
                   required=True)
    p.add_argument("--n_seeds", type=int, default=5,
                   help="Number of independent seed-runs to average (paper default 5).")
    p.add_argument("--n_train", type=int, default=None,
                   help="Override train-set size (default 5000 from config).")
    p.add_argument("--n_test", type=int, default=None,
                   help="Override test-set size (default 5000 from config).")
    p.add_argument("--max_iters", type=int, default=None,
                   help="Override max_iters for quick smoke testing.")
    p.add_argument("--device", default=None,
                   help="cpu or cuda; default from config (cpu).")
    p.add_argument("--seed_start", type=int, default=0,
                   help="First seed index (default 0). Each seed s also "
                        "generates the test split with seed (s+100).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-log_every progress.")
    args = p.parse_args()

    saved = load_config(args.dataset)
    data_alpha = saved.get("data_alpha", 1.0)
    n_train = args.n_train if args.n_train else saved["n_train"]
    n_test  = args.n_test  if args.n_test  else saved["n_test"]
    overrides = {}
    if args.max_iters is not None:
        overrides["max_iters"] = args.max_iters
    if args.device is not None:
        overrides["device"] = args.device
    train_cfg = build_train_config(saved, overrides)

    print(f"=== VIBMask demo on {args.dataset.upper()} ===")
    print(f"Config: {CONFIG_DIR / (args.dataset + '.json')}")
    print(f"  data_alpha={data_alpha}, n_train={n_train}, n_test={n_test}, "
          f"n_seeds={args.n_seeds}")
    print(f"  K={train_cfg.num_selectors}, gate={train_cfg.gate_type}, "
          f"max_iters={train_cfg.max_iters}, lr={train_cfg.learning_rate:g}")
    print(f"  β={train_cfg.beta:g}, γ={train_cfg.gamma:g}, σ={train_cfg.sigma}")
    print()

    per_seed = []
    for s in range(args.seed_start, args.seed_start + args.n_seeds):
        print(f"--- seed {s} ---")
        fm = run_one_seed(args.dataset, s, train_cfg,
                          data_alpha, n_train, n_test,
                          saved.get("soft_inference", False), args.verbose)
        print(f"  {summarise(fm)}  (train_time={fm['train_time_s']:.1f}s)")
        per_seed.append(fm)

    # Aggregate.
    print()
    print(f"=== {args.dataset.upper()} summary over {len(per_seed)} seeds ===")
    is_conditional = args.dataset.lower() in {"syn4", "syn5", "syn6"}
    keys = ["tpr_mean", "fdr_mean", "acc"]
    if is_conditional:
        keys = ["cfsr"] + keys
    for k in keys:
        vals = np.array([m[k] for m in per_seed])
        label = {"tpr_mean": "TPR", "fdr_mean": "FDR",
                 "cfsr": "CFSR", "acc": "ACC"}[k]
        if k == "acc":
            print(f"  {label:<5} = {vals.mean():.4f} ± {vals.std():.4f}")
        else:
            print(f"  {label:<5} = {vals.mean():.2f} ± {vals.std():.2f}")


if __name__ == "__main__":
    main()
