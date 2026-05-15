"""VIBMask: Instance-Wise Feature Selection via Variational Information Bottleneck.

Reference implementation accompanying the IJCAI-ECAI 2026 paper
"Learning Local Feature Masks with Variational Information Bottleneck".

Public API:
    SynDataset, generate_synthetic   - synthetic Syn1-Syn6 datasets (paper §5.1)
    VIBMask, Selector, Predictor     - model components (paper §3)
    TrainConfig, train_vibmask       - training loop
    predict_with_masks               - inference helper returning predictions + masks
    feature_metrics, accuracy        - TPR / FDR / CFSR / ACC metrics (Appendix A6.4)
"""

from .data import SynDataset, generate_synthetic
from .loss import (
    bootstrap_weights,
    compute_feature_weights,
    gate_regularizer,
    gaussian_cdf,
)
from .metrics import accuracy, feature_metrics, summarise
from .model import Predictor, Selector, VIBMask
from .train import TrainConfig, TrainHistory, predict_with_masks, train_vibmask

__all__ = [
    # data
    "SynDataset", "generate_synthetic",
    # model
    "VIBMask", "Selector", "Predictor",
    # training
    "TrainConfig", "TrainHistory", "train_vibmask", "predict_with_masks",
    # loss
    "compute_feature_weights", "gate_regularizer", "bootstrap_weights", "gaussian_cdf",
    # metrics
    "accuracy", "feature_metrics", "summarise",
]
__version__ = "1.0.0"
