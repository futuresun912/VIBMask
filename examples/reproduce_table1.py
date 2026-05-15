"""Reproduce Table 1 of the paper end-to-end: runs all six synthetic datasets
(Syn1-Syn6) with the released best-config and prints a markdown-formatted
table mirroring the paper's reported numbers.

Default: 5 seeds per dataset (paper protocol). Total wall time on a single
CPU core: about 30-60 minutes depending on machine.

Example
-------
    python examples/reproduce_table1.py                 # full run, 5 seeds
    python examples/reproduce_table1.py --n_seeds 1     # quick smoke (5x faster)
    python examples/reproduce_table1.py --max_iters 2000  # even quicker smoke
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from vibmask import (
    TrainConfig,
    feature_metrics,
    generate_synthetic,
    predict_with_masks,
    train_vibmask,
)


CONFIG_DIR = Path(__file__).parent.parent / "configs"
SYN_DATASETS = [f"syn{i}" for i in range(1, 7)]


def load_config(dataset: str) -> dict:
    with open(CONFIG_DIR / f"{dataset}.json") as fh:
        return json.load(fh)


def build_train_config(saved: dict, max_iters: int | None,
                       device: str | None) -> TrainConfig:
    tc = dict(saved["train_config"])
    tc["sel_hidden"]  = tuple(tc["sel_hidden"])
    tc["pred_hidden"] = tuple(tc["pred_hidden"])
    if max_iters is not None: tc["max_iters"] = max_iters
    if device is not None:    tc["device"]    = device
    return TrainConfig(**tc)


def run_dataset(dataset: str, n_seeds: int, max_iters: int | None,
                device: str | None):
    saved = load_config(dataset)
    train_cfg = build_train_config(saved, max_iters, device)
    data_alpha = saved.get("data_alpha", 1.0)
    soft = saved.get("soft_inference", False)
    n_train = saved["n_train"]; n_test = saved["n_test"]

    per_seed = []
    t_start = time.time()
    for s in range(n_seeds):
        train_ds = generate_synthetic(dataset, n=n_train, seed=s, alpha=data_alpha)
        test_ds  = generate_synthetic(dataset, n=n_test,  seed=s + 100, alpha=data_alpha)
        model, _, _ = train_vibmask(
            train_ds.X, train_ds.y, output_dim=2,
            config=replace(train_cfg, seed=s),
        )
        pred = predict_with_masks(model, test_ds.X,
                                  device=train_cfg.device, hard=not soft)
        cfi = 10 if dataset in {"syn4", "syn5", "syn6"} else None
        fm = feature_metrics(test_ds.gt, pred["mask"], control_flow_idx=cfi)
        fm["acc"] = float((pred["preds"] == test_ds.y).mean())
        per_seed.append(fm)
    return per_seed, time.time() - t_start


def aggregate(per_seed: list[dict], dataset: str) -> dict:
    out = {}
    is_conditional = dataset in {"syn4", "syn5", "syn6"}
    for k in ("tpr_mean", "fdr_mean", "acc"):
        vals = np.array([m[k] for m in per_seed])
        out[f"{k}_mean_over_seeds"] = float(vals.mean())
        out[f"{k}_std_over_seeds"]  = float(vals.std())
    if is_conditional:
        vals = np.array([m["cfsr"] for m in per_seed])
        out["cfsr_mean_over_seeds"] = float(vals.mean())
        out["cfsr_std_over_seeds"]  = float(vals.std())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--max_iters", type=int, default=None,
                   help="Override max_iters (default: per-dataset config). "
                        "Set 2000 for a quick smoke run.")
    p.add_argument("--device", default=None,
                   help="Override training device (default: per-dataset config).")
    p.add_argument("--out_json", default=None,
                   help="Optional JSON path to dump full results.")
    args = p.parse_args()

    all_results = {}
    print(f"Running VIBMask on {len(SYN_DATASETS)} synthetic datasets "
          f"with {args.n_seeds} seed(s) each...")
    print()

    for ds in SYN_DATASETS:
        print(f"--- {ds.upper()} ---")
        t0 = time.time()
        per_seed, wall = run_dataset(ds, args.n_seeds, args.max_iters, args.device)
        agg = aggregate(per_seed, ds)
        all_results[ds] = {"per_seed": per_seed, "summary": agg, "wall_seconds": wall}
        if ds in {"syn4", "syn5", "syn6"}:
            print(f"  TPR={agg['tpr_mean_mean_over_seeds']:.2f}±"
                  f"{agg['tpr_mean_std_over_seeds']:.2f}  "
                  f"FDR={agg['fdr_mean_mean_over_seeds']:.2f}±"
                  f"{agg['fdr_mean_std_over_seeds']:.2f}  "
                  f"CFSR={agg['cfsr_mean_over_seeds']:.2f}±"
                  f"{agg['cfsr_std_over_seeds']:.2f}  "
                  f"ACC={agg['acc_mean_over_seeds']:.4f}±"
                  f"{agg['acc_std_over_seeds']:.4f}  "
                  f"({wall:.0f}s)")
        else:
            print(f"  TPR={agg['tpr_mean_mean_over_seeds']:.2f}±"
                  f"{agg['tpr_mean_std_over_seeds']:.2f}  "
                  f"FDR={agg['fdr_mean_mean_over_seeds']:.2f}±"
                  f"{agg['fdr_mean_std_over_seeds']:.2f}  "
                  f"ACC={agg['acc_mean_over_seeds']:.4f}±"
                  f"{agg['acc_std_over_seeds']:.4f}  "
                  f"({wall:.0f}s)")

    # Final markdown table.
    print()
    print("===== Final Table (paper Table 1, VIBMask row) =====")
    print()
    print("| dataset | TPR ↑ | FDR ↓ | CFSR ↑ | ACC ↑ |")
    print("|---------|------:|------:|-------:|------:|")
    for ds in SYN_DATASETS:
        a = all_results[ds]["summary"]
        cfsr = (f"{a['cfsr_mean_over_seeds']:.1f}" if ds in {"syn4","syn5","syn6"}
                else "  -  ")
        print(f"| {ds.upper():<7} | "
              f"{a['tpr_mean_mean_over_seeds']:5.1f} | "
              f"{a['fdr_mean_mean_over_seeds']:5.1f} | "
              f"{cfsr:>5} | "
              f"{a['acc_mean_over_seeds']:.3f} |")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nDumped full results to {args.out_json}")


if __name__ == "__main__":
    main()
