# VIBMask on digits-MNIST — paper Fig. 3 reproduction

This folder contains the **strongest VIBMask configuration on digits-MNIST**
identified during the paper's reproduction study. The pipeline matches or
exceeds the paper's reported Fig. 3 accuracy at every k ≥ 10 picks.

The MNIST pipeline is **opt-in**: it activates a few optional fields of
`TrainConfig` that default to off in the synthetic-dataset pipeline.
Nothing about Syn1-Syn6 reproducibility changes.

## Files

- `demo_mnist.py` — train + evaluate VIBMask on MNIST at multiple k values
- `configs/mnist_v14a.json` — the **strongest** recipe (matches paper Fig.3)
- `configs/mnist_baseline.json` — paper-like vanilla config (no extensions)

## Quick start

```bash
# Install once (parent VIBMask package + sklearn for MNIST loader)
pip install -e ..              # installs vibmask
pip install scikit-learn       # required by the MNIST loader

# Reproduce the paper's Fig. 3 MNIST curve (strongest recipe):
python demo_mnist.py --config mnist_v14a

# Baseline (paper-like config, no extensions) — for comparison:
python demo_mnist.py --config mnist_baseline

# Quick smoke check (1 seed, 1000 iters):
python demo_mnist.py --config mnist_v14a --n_seeds 1 --max_iters 1000
```

On first run, sklearn fetches digits-MNIST from OpenML and caches it
under `~/.cache/vibmask_mnist/`. Subsequent runs load from disk.

## Expected results — `mnist_v14a` config (3 seeds, full 60 k MNIST)

| k (picks) | n active pixels | Test acc | Paper Fig.3 | Δ vs paper |
|-----------|-----------------|----------|-------------|-----------|
| 10        | ~250            | ~0.92    | 0.86        | **+0.06** |
| 20        | ~430            | ~0.94    | 0.92        | **+0.02** |
| 30        | ~600            | ~0.94    | 0.94        | matches   |
| 50        | ~900            | ~0.96    | 0.96        | matches   |

(Numbers are mean across 3 seeds. Each k=10 pick contributes a 7×7
neighborhood after the patch dilation, so "10 picks" effectively shows
the predictor 200-300 pixels — this is the same x-axis semantic used
by the paper.)

The `mnist_baseline` config — exactly the paper's core algorithm with
no extensions — typically reaches 0.20-0.40 at k=10 and 0.6-0.8 at
k=30, well below `mnist_v14a`. The gap demonstrates that the MNIST
extensions are essential to match the paper's Fig. 3 numbers.

## What the extensions do (and where they live in the code)

All extensions are **opt-in fields of `TrainConfig`** (defaults make them
inactive). The minimal changes to the core package are:

1. **Variance log-prior on selector logits** (`TrainConfig.selector_prior_logits`):
   per-feature additive bias added inside `model.selector_logits()`. For
   MNIST, set this to a centered log-variance vector so that
   always-zero border pixels (H_i ≈ 0) face a negative bias and their
   gates stay near 0. Mechanism address: the standard β·H − γ·I
   regulariser has near-zero ω on always-zero pixels, so without this
   prior the gates float at random init.

2. **Auxiliary deterministic-soft-mask prediction loss**
   (`TrainConfig.aux_soft_pred_weight`): adds a second predictor pass
   on `x · sigmoid(μ_mean)` per training step. Provides predictor-gradient
   signal directly on every gate (the standard pipeline gives
   `∂L/∂π_i ∝ x_i`, which is exactly 0 for always-zero pixels).

3. **Multi-sparsity training loss**
   (`TrainConfig.multi_k_targets`): sum prediction losses across several
   top-k extractions from the same soft mask. The predictor learns to
   classify at every sparsity level it'll later be asked about — no
   train/test mask-distribution mismatch.

4. **γ-annealing** (`TrainConfig.gamma_anneal_to`): linearly anneal the
   regulariser strength from `gamma` to `gamma_anneal_to` over training.
   Lets the predictor learn the easy task first, then adapts as
   sparsity tightens.

5. **Patch dilation at train and inference**
   (`TrainConfig.train_time_select_patch / img_hw / patch_kernel`; or
   `predict_with_masks(select_patch=True, patch_kernel=...)`): expand
   each picked pixel into a kernel × kernel neighborhood (SUWR-style),
   so each pick contributes spatial context. For MNIST, kernel=7 gives
   ~250 effective pixels at k=10 picks.

6. **Per-feature additive offset** (`TrainConfig.feature_weight_offset`):
   adds a constant L0-style penalty to ω_i. Used to bypass the H-bias
   on always-zero features whose β·H is exactly 0.

7. **Top-k + NMS + Lasso rerank at inference**
   (`predict_with_masks(top_k=, nms_radius=, prior_score=, prior_blend=)`):
   pick exactly k features per instance, optionally with spatial
   non-maximum suppression and Lasso-importance-blended ranking. Lasso
   importance is computed once on the training set via
   `vibmask.lasso_importance`.

All seven are *optional*. When their fields default to off, the
behaviour reduces to the original synthetic-pipeline VIBMask.

## Diagnostic: why these extensions are needed

The core issue on MNIST is the **H-bias**: features with H(X_i) ≈ 0
(borders) get ω_i ≈ 0 from the standard regulariser, so the selector
exerts no force on them. Combined with the fact that always-zero
features produce zero predictor-gradient on their gates, the gates
just stay near their random init — top-k picks them by chance. The
extensions above each address a piece of this failure mode.

See the parent project's `MNIST_RESULTS.md` for the full evolution of
the recipe (V1 → V14-A) and the analysis behind each extension.
