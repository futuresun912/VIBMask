"""Training loop for VIBMask.

Implements paper Algorithm 1 with bootstrap-based selector bagging:
  - predictor sees X · M̃ (M̃ = mean of K soft selector gates)
  - each selector's regulariser is computed under its own per-batch
    bootstrap weighting, giving the bagging effect without a K× memory cost.

The trainer is intentionally framework-light (pure PyTorch + NumPy) so it
can serve as a smoke-tested reference implementation rather than a feature
race against PyTorch Lightning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .loss import bootstrap_weights, compute_feature_weights, gate_regularizer
from .model import VIBMask


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ---- data ----
    batch_size: int = 64
    val_fraction: float = 0.1
    # ---- optimisation ----
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_iters: int = 10_000
    early_stop_patience: int = 500
    # ---- VIBMask model ----
    num_selectors: int = 3
    sigma: float = 0.5
    train_sigma: bool = False
    sel_hidden: tuple = (100, 100, 100)
    pred_hidden: tuple = (100, 100, 10)
    layer_norm: bool = False
    activation: str = "leaky_relu"
    pred_dropout: float = 0.0
    # ---- regulariser (β: feature-entropy weight, γ: MI weight; paper Theorem 1) ----
    beta: float = 1e-4
    gamma: float = 1e-4
    # iters with β=γ=0 before the regulariser kicks in. 0 disables.
    warmup_iters: int = 0
    # ---- gate parameterisation ----
    # "gaussian_clip" = clip(μ + σ·ε, 0, 1) (paper default); "hard_concrete" =
    # Louizos-style HC distribution (true 0/1 mass).
    gate_type: str = "gaussian_clip"
    hc_beta: float = 0.5  # hard-concrete temperature
    # ---- training stabilisers ----
    ema_decay: float = 0.0          # EMA decay for parameters; 0 disables.
    lr_schedule: str = "constant"   # "constant" or "cosine" (anneal to lr*0.01).
    label_smoothing: float = 0.0    # cross-entropy label smoothing.
    grad_clip: float = 1.0
    bootstrap: bool = True          # per-batch multinomial bootstrap (Alg 1 line 4)
    # ---- misc ----
    classification: bool = True
    log_every: int = 200
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainHistory:
    iters: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[DataLoader, DataLoader]:
    n = X_train.shape[0]
    n_val = max(1, int(round(n * val_fraction)))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def to_tensor_set(idx):
        return TensorDataset(
            torch.from_numpy(X_train[idx]).float(),
            torch.from_numpy(y_train[idx]).long(),
        )

    train_loader = DataLoader(
        to_tensor_set(tr_idx), batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        to_tensor_set(val_idx),
        batch_size=max(batch_size, len(val_idx)),
        shuffle=False,
    )
    return train_loader, val_loader


def _classification_loss(
    logits: torch.Tensor, y: torch.Tensor, output_dim: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if output_dim == 1:
        return nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), y.float()
        )
    return nn.functional.cross_entropy(logits, y,
                                       label_smoothing=float(label_smoothing))


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits.dim() == 2 and logits.shape[1] > 1:
        pred = logits.argmax(dim=1)
    else:
        pred = (logits.squeeze(-1) > 0).long()
    return float((pred == y).float().mean().item())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_vibmask(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dim: int,
    config: TrainConfig | None = None,
    progress_cb: Callable[[int, dict], None] | None = None,
) -> Tuple[VIBMask, np.ndarray, TrainHistory]:
    """Train a VIBMask model on (X_train, y_train).

    Returns the trained model, the precomputed feature weights ω, and a
    history object with per-log iteration losses/accuracy.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    # 1. Feature weights ω = β·H − γ·I (paper Theorem 1).
    weights_np = compute_feature_weights(X_train, y_train, beta=cfg.beta, gamma=cfg.gamma)
    weights = torch.from_numpy(weights_np).float().to(device)

    # 2. Data.
    train_loader, val_loader = _make_loaders(
        X_train, y_train, cfg.val_fraction, cfg.batch_size, rng
    )

    # 3. Model & optimiser.
    activation_cls = {
        "leaky_relu": nn.LeakyReLU, "relu": nn.ReLU, "tanh": nn.Tanh,
    }[cfg.activation]
    model = VIBMask(
        input_dim=X_train.shape[1],
        output_dim=output_dim,
        num_selectors=cfg.num_selectors,
        sigma=cfg.sigma,
        sel_hidden=cfg.sel_hidden,
        pred_hidden=cfg.pred_hidden,
        train_sigma=cfg.train_sigma,
        layer_norm=cfg.layer_norm,
        activation=activation_cls,
        gate_type=cfg.gate_type,
        hc_beta=cfg.hc_beta,
        pred_dropout=cfg.pred_dropout,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, cfg.max_iters), eta_min=cfg.learning_rate * 0.01
        )

    # EMA shadow parameters.
    ema_decay = float(cfg.ema_decay)
    ema_state: dict[str, torch.Tensor] | None = None
    if ema_decay > 0.0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 4. Training loop.
    history = TrainHistory()
    best_val = float("inf")
    best_state = None
    bad_iters = 0
    iter_idx = 0
    train_iter = _infinite_loader(train_loader)

    while iter_idx < cfg.max_iters:
        model.train()
        batch = next(train_iter)
        x, y = batch[0].to(device), batch[1].to(device)

        out = model(x)
        loss_pred = _classification_loss(out["logits"], y, output_dim,
                                         label_smoothing=cfg.label_smoothing)

        sw = (bootstrap_weights(x.shape[0], cfg.num_selectors, device=device)
              if cfg.bootstrap else None)
        loss_reg = gate_regularizer(
            mus=out["mus"], weights=weights, sigma=model.sigma, sample_weights=sw,
        )

        # Warmup: scale the regulariser to 0 for the first warmup_iters,
        # then ramp linearly over the next 1k iters to full strength.
        if cfg.warmup_iters > 0:
            ramp = 1000
            if iter_idx < cfg.warmup_iters:
                w_factor = 0.0
            elif iter_idx < cfg.warmup_iters + ramp:
                w_factor = (iter_idx - cfg.warmup_iters) / ramp
            else:
                w_factor = 1.0
        else:
            w_factor = 1.0

        loss = loss_pred + w_factor * loss_reg
        optim.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        if scheduler is not None:
            scheduler.step()

        # EMA update.
        if ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                    else:
                        ema_state[k].copy_(v)

        if iter_idx % cfg.log_every == 0 or iter_idx == cfg.max_iters - 1:
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
                val_loss, val_acc = _evaluate(model, val_loader, weights, output_dim, device)
                model.load_state_dict(live_state)
            else:
                val_loss, val_acc = _evaluate(model, val_loader, weights, output_dim, device)
            history.iters.append(iter_idx)
            history.train_loss.append(float(loss.item()))
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            if progress_cb is not None:
                progress_cb(iter_idx, {
                    "train_loss": float(loss.item()),
                    "train_pred": float(loss_pred.item()),
                    "train_reg": float(loss_reg.item()),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                snap_src = ema_state if ema_state is not None else model.state_dict()
                best_state = {k: v.detach().cpu().clone() for k, v in snap_src.items()}
                bad_iters = 0
            else:
                bad_iters += cfg.log_every
                if bad_iters >= cfg.early_stop_patience:
                    break
        iter_idx += 1

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, weights_np, history


def _infinite_loader(loader: DataLoader) -> Iterable:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _evaluate(
    model: VIBMask,
    loader: DataLoader,
    weights: torch.Tensor,
    output_dim: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses, accs, n = 0.0, 0.0, 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        loss_pred = _classification_loss(out["logits"], y, output_dim)
        loss_reg = gate_regularizer(out["mus"], weights, sigma=model.sigma)
        bs = x.shape[0]
        losses += float((loss_pred + loss_reg).item()) * bs
        accs += _accuracy(out["logits"], y) * bs
        n += bs
    if n == 0:
        return 0.0, 0.0
    return losses / n, accs / n


@torch.no_grad()
def predict_with_masks(
    model: VIBMask,
    X: np.ndarray,
    device: torch.device | str = "cpu",
    hard: bool = True,
    batch_size: int = 1024,
) -> dict:
    """Run inference and return predictions + per-sample binary masks.

    Returns a dict with:
      - 'logits':     [N, output_dim] numpy
      - 'preds':      [N] integer predictions
      - 'mask':       [N, d] binary mask (majority vote when hard=True)
      - 'soft_mask':  [N, d] continuous M̃ (no noise)
      - 'hard_mask':  [N, d] strict majority-vote binary mask
    """
    device = torch.device(device)
    model = model.to(device).eval()
    n = X.shape[0]
    out_logits, out_mask, out_soft, out_hard = [], [], [], []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.from_numpy(X[start:end]).float().to(device)
        out = model.predict(x, hard=hard)
        out_logits.append(out["logits"].cpu().numpy())
        out_mask.append(out["mask"].cpu().numpy())
        out_soft.append(out["soft_mask"].cpu().numpy())
        out_hard.append(out["hard_mask"].cpu().numpy())
    logits = np.concatenate(out_logits, axis=0)
    mask = np.concatenate(out_mask, axis=0)
    soft = np.concatenate(out_soft, axis=0)
    hard_mask = np.concatenate(out_hard, axis=0)
    if logits.shape[1] == 1:
        preds = (logits.squeeze(-1) > 0).astype(np.int64)
    else:
        preds = logits.argmax(axis=1).astype(np.int64)
    return {"logits": logits, "preds": preds, "mask": mask,
            "soft_mask": soft, "hard_mask": hard_mask}
