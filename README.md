# VIBMask

**Learning Local Feature Masks with Variational Information Bottleneck**
_IJCAI-ECAI 2026_

VIBMask is an instance-wise feature selection (IWFS) framework that learns
per-sample binary masks `M ∈ {0, 1}^d` over the input features so that
prediction can be made from only a small, semantically meaningful subset.
The masks are produced by an ensemble of `K` MLP selectors and refined via
a variational lower bound on the information-bottleneck objective.

This repository contains the reference implementation for the **synthetic
datasets (Syn1–Syn6)** experiments reported in §5.1 / Table 1 of the
paper. Real-world experiment code is intentionally kept out of this
release; please refer to the paper's technical appendix for that
description.

---

## Contents

```
vibmask/
├── vibmask/                   # core package
│   ├── __init__.py            # public API
│   ├── data.py                # Syn1–Syn6 dataset generator
│   ├── loss.py                # ω = β·H − γ·I feature weights, gate regulariser
│   ├── metrics.py             # TPR / FDR / CFSR / ACC
│   ├── model.py               # Selector + Predictor + VIBMask classes
│   └── train.py               # TrainConfig + train_vibmask + predict_with_masks
├── configs/                   # per-dataset best hyperparameters
│   ├── syn1.json
│   ├── syn2.json
│   ├── syn3.json
│   ├── syn4.json
│   ├── syn5.json
│   └── syn6.json
├── examples/
│   ├── demo_synthetic.py      # run VIBMask on one dataset (CLI)
│   └── reproduce_table1.py    # run all six datasets, print Table 1
├── tests/                     # smoke tests
├── pyproject.toml             # package metadata + dependencies
├── requirements.txt           # pip dependency list
├── environment.yml            # conda environment file
├── LICENSE                    # MIT
└── README.md
```

## Installation

VIBMask requires **Python ≥ 3.9** and **PyTorch ≥ 2.0**. CPU is sufficient
for all synthetic experiments; the package runs on GPU without changes by
setting `device="cuda"` in the config.

### Option 1 — pip (recommended)

```bash
pip install -r requirements.txt
pip install -e .             # installs the `vibmask` package
```

### Option 2 — conda

```bash
conda env create -f environment.yml
conda activate vibmask
pip install -e .
```

### Verify installation

```bash
pytest -q                    # runs the smoke tests
```

## Quick start

### Run one synthetic dataset

```bash
python examples/demo_synthetic.py --dataset syn1
```

This trains VIBMask on Syn1 with the released best hyperparameters (see
`configs/syn1.json`), repeats over 5 seeds, and prints per-seed and
aggregated TPR / FDR / ACC. Expected output:

```
=== VIBMask demo on SYN1 ===
Config: configs/syn1.json
  data_alpha=5.0, n_train=5000, n_test=5000, n_seeds=5
  K=4, gate=hard_concrete, max_iters=10000, lr=1.848e-05
  β=0.001, γ=0, σ=0.891

--- seed 0 ---
  acc=0.812 TPR=100.0 FDR=0.0  (train_time=33.4s)
…
=== SYN1 summary over 5 seeds ===
  TPR   = 100.00 ± 0.00
  FDR   = 0.00 ± 0.00
  ACC   = 0.8091 ± 0.0058
```

### Reproduce Table 1 (all six datasets)

```bash
python examples/reproduce_table1.py
```

Default settings (5 seeds × 6 datasets) take ~30–60 minutes on a single
modern CPU core. The script prints a markdown-formatted table at the end
that matches the VIBMask row of the paper's Table 1.

For a faster sanity check:

```bash
python examples/reproduce_table1.py --n_seeds 1 --max_iters 2000
```

## Expected results

Running `reproduce_table1.py` with default settings reproduces the
following published numbers within seed-to-seed variance:

| dataset | TPR ↑ | FDR ↓ | CFSR ↑ | ACC ↑ |
|---------|------:|------:|-------:|------:|
| Syn1    | 100.0 |   0.0 |    -   | 0.809 |
| Syn2    | 100.0 |   0.0 |    -   | 0.903 |
| Syn3    | 100.0 |   0.0 |    -   | 0.950 |
| Syn4    |  99.1 |   4.5 |  100.0 | 0.831 |
| Syn5    |  91.6 |   2.1 |   91.3 | 0.852 |
| Syn6    |  99.4 |  15.0 |   99.9 | 0.951 |

Numbers above are the mean over 5 independent seeds with `n_train =
n_test = 5000` and the per-dataset configurations under `configs/`.

## Method summary

For each input `X ∈ R^d`, an ensemble of `K` MLP selectors
`g_{ψ_1}, …, g_{ψ_K}` produces continuous logits `μ_k ∈ R^d`. The
relaxed-Bernoulli gate

```
M̃_k = clip(μ_k + σ · ε, 0, 1),   ε ~ N(0, I)
```

is averaged across selectors `M̃ = (1/K) Σ_k M̃_k`, and the predictor
`f_θ` operates on the gated input `f_θ(X ⊙ M̃)`. At inference (`ε = 0`)
the mask is binarised by majority vote over the `K` selectors.

The training objective combines the prediction loss with a closed-form
Gaussian-CDF regulariser

```
L = CE(y, f_θ(X ⊙ M̃)) + (1/K) Σ_k mean_n Σ_i ω_i · Φ(μ_{k,n,i} / σ)
```

where `ω_i = β · H(X_i) − γ · I(X_i; Y)` are precomputed per-feature
weights (paper Theorem 1). Selectors are trained with a per-batch
multinomial bootstrap that gives an `O(B)` bagging effect without
materialising `K` parallel minibatches.

Two gate parameterisations are supported (`gate_type` in `TrainConfig`):

| `gate_type`       | distribution                                       |
|-------------------|----------------------------------------------------|
| `gaussian_clip`   | `clip(μ + σ · ε, 0, 1)` (paper default)            |
| `hard_concrete`   | Louizos 2018 hard-concrete with stretching β       |

Optional training stabilisers (all opt-in via `TrainConfig`): EMA shadow
parameters, cosine LR schedule, regulariser warmup, and label smoothing.

## Public API

```python
from vibmask import (
    SynDataset, generate_synthetic,      # data
    VIBMask, Selector, Predictor,        # model
    TrainConfig, train_vibmask, predict_with_masks,  # training & inference
    feature_metrics, accuracy,           # metrics
)
```

Minimal usage:

```python
from vibmask import TrainConfig, generate_synthetic, train_vibmask, predict_with_masks

train = generate_synthetic("syn5", n=5000, seed=0, alpha=5.0)
test  = generate_synthetic("syn5", n=5000, seed=100, alpha=5.0)

cfg = TrainConfig(
    num_selectors=3, sigma=0.5,
    sel_hidden=(100, 100, 100), pred_hidden=(100, 100, 10),
    beta=1e-4, gamma=1e-4, gate_type="gaussian_clip",
    max_iters=5000, learning_rate=1e-3, batch_size=64, seed=0,
)
model, weights, history = train_vibmask(train.X, train.y, output_dim=2, config=cfg)
pred = predict_with_masks(model, test.X, hard=True)
print("test_acc =", (pred["preds"] == test.y).mean())
print("avg #features selected =", pred["mask"].sum(axis=1).mean())
```

## Configuration file format

Each `configs/synK.json` contains the best hyperparameters discovered by
the paper's tuning sweep, together with documentation of the expected
metrics. The schema is:

```jsonc
{
  "_documentation": "...",        // human-readable description
  "dataset": "syn1",
  "data_alpha": 5.0,              // sigmoid sharpness used during data generation
  "n_train": 5000,
  "n_test":  5000,
  "n_seeds": 5,
  "train_config": { ... },        // TrainConfig kwargs
  "soft_inference": false         // hard (majority-vote) vs. soft (mean) mask
}
```

You can override any of these fields on the command line for
`demo_synthetic.py` and `reproduce_table1.py` — see `--help`.

## Reproducibility notes

1. **Data-temperature α.** The original INVASE/L2X supplementary formula
   uses `α = 1` for the synthetic logit, which makes the labels noisy
   (Bayes-optimal accuracy ≈ 0.63 on Syn1, ≈ 0.82 on Syn2/3). The
   accuracies reported in the paper require sharpening the sigmoid via
   `α ∈ {2, 3, 5}` (per-dataset values are stored in each
   `configs/synK.json`). All experiments in this release set `α`
   explicitly.

2. **Determinism.** Each seed `s` reproduces both train (`seed = s`) and
   test (`seed = s + 100`) data exactly via NumPy's `default_rng`. We
   also call `torch.manual_seed` at the start of `train_vibmask`. Note
   that some PyTorch ops (`F.cross_entropy` with `label_smoothing`)
   are non-deterministic on certain GPU backends; CPU runs are exactly
   reproducible across machines.

3. **Hardware sensitivity.** All published numbers were produced on
   CPU (AMD EPYC 9455P, single thread per process). Switching to GPU
   may shift accuracies by ≤ 0.5 pp due to non-determinism in CUDA
   kernels.

## Citation

```bibtex
@inproceedings{vibmask2026,
  title     = {Learning Local Feature Masks with Variational Information Bottleneck},
  author    = {VIBMask authors},
  booktitle = {Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
