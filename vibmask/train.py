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
    # ---- OPTIONAL EXTENSIONS (all default to off ⇒ original behaviour unchanged) ----
    # See `examples/mnist/` for the "strongest VIBMask on MNIST" recipe.
    #
    # Per-feature additive offset on ω_i applied AFTER β·H − γ·I. Accepts
    # scalar OR length-d tuple. Used for the H-bias bypass on data where
    # some features have H_i ≈ 0 (e.g. MNIST border pixels): a positive
    # offset gives every feature a uniform L0-style penalty.
    feature_weight_offset: object = 0.0
    # Linear interpolation of γ over the post-warmup window. When set, the
    # regulariser weights ramp from cfg.gamma → gamma_anneal_to so the
    # predictor first sees a dense gate (γ_start large) and adapts as
    # sparsity tightens.
    gamma_anneal_to: float | None = None
    # Length-d tensor added as a constant bias to every selector's μ
    # output. Use this to (a) warm-start the selector from a feature
    # prior (e.g. Lasso coefficients or log-variance) — small magnitude
    # like (imp - mean)·2; or (b) HARD-PRUNE useless features by setting
    # very negative values (~-1000) on them so their gate is pinned to ≈0.
    selector_prior_logits: tuple = ()
    # Auxiliary prediction loss using the deterministic soft mask
    # (sigmoid(μ_mean) directly) as predictor input. Adds α·CE to the
    # loss. Gives the selector direct gradient signal on every gate,
    # essential for features where x_i ≈ 0 (predictor gradient is
    # otherwise exactly zero). 0 disables.
    aux_soft_pred_weight: float = 0.0
    # Multi-sparsity training: when set to a non-empty tuple of k values,
    # at each step the predictor loss is summed over k top-k mask
    # extractions (plus the original soft mask). Use 0 in the tuple to
    # mean "the full untruncated soft mask". Predictor learns to predict
    # accurately at every sparsity level it'll be asked about.
    multi_k_targets: tuple = ()
    # When true, the training-time top-k path (used by multi_k_targets)
    # dilates each picked feature into a `train_time_patch_kernel`-sized
    # 2D neighborhood. Requires `train_time_img_hw` so the flat mask can
    # be reshaped to 2D. For MNIST pixels: img_hw=28.
    train_time_select_patch: bool = False
    train_time_img_hw: int = 0
    train_time_patch_kernel: int = 3
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


def _patch_dilate(mask: torch.Tensor, img_hw: int, kernel: int = 3) -> torch.Tensor:
    """Dilate a flat binary mask by a kernel×kernel block per active pixel.

    Used by the optional `train_time_select_patch` path and by
    `predict_with_masks(select_patch=True)`. Mirrors how SUWR (Oosterhuis
    et al. 2025) reports image experiments — each selected pixel anchors
    a kernel×kernel neighborhood. For MNIST: img_hw=28.
    """
    B, d = mask.shape
    if d != img_hw * img_hw:
        raise ValueError(f"_patch_dilate: d={d} != img_hw²={img_hw*img_hw}")
    m2 = mask.view(B, 1, img_hw, img_hw)
    pad = kernel // 2
    m2 = torch.nn.functional.max_pool2d(m2, kernel_size=kernel, stride=1, padding=pad)
    return m2.view(B, d)


def _topk_nms(scores: torch.Tensor, k: int, img_hw: int, radius: int) -> torch.Tensor:
    """Spatial non-maximum-suppression top-k pick.

    Picks the highest-scoring location, suppresses a (2r+1)×(2r+1)
    neighborhood, repeats k times. Forces spatial diversity so each pick
    contributes independent local context. Used by
    `predict_with_masks(nms_radius>0)`. Returns a [B, d] binary mask.
    """
    B, d = scores.shape
    if d != img_hw * img_hw:
        raise ValueError(f"_topk_nms: d={d} != img_hw²={img_hw*img_hw}")
    if radius <= 0:
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)
        out = torch.zeros_like(scores)
        out.scatter_(1, topk_idx, 1.0)
        return out
    s = scores.clone()
    mask = torch.zeros_like(scores)
    batch_idx = torch.arange(B, device=scores.device)
    offsets = [(dr, dc) for dr in range(-radius, radius + 1)
                        for dc in range(-radius, radius + 1)]
    for _ in range(k):
        idx = s.argmax(dim=1)
        cur_score = s[batch_idx, idx]
        active = cur_score > float('-inf')
        if active.any():
            picks = idx.clone()
            mask[batch_idx[active], picks[active]] = 1.0
            rows = picks // img_hw
            cols = picks % img_hw
            for dr, dc in offsets:
                rr = (rows + dr).clamp(0, img_hw - 1)
                cc = (cols + dc).clamp(0, img_hw - 1)
                nb_idx = rr * img_hw + cc
                s[batch_idx[active], nb_idx[active]] = float('-inf')
        if not active.any():
            break
    return mask


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
    # Optional: γ annealing precomputes the end-of-training weights so the
    # main loop can linearly interpolate per-iter (ω is linear in (β, γ)).
    anneal_active = cfg.gamma_anneal_to is not None
    if anneal_active:
        weights_end_np = compute_feature_weights(
            X_train, y_train, beta=cfg.beta, gamma=float(cfg.gamma_anneal_to)
        )
    else:
        weights_end_np = None
    # Optional: per-feature additive offset (scalar or length-d).
    off = cfg.feature_weight_offset
    if isinstance(off, (tuple, list, np.ndarray)) and len(off) > 0:
        off_np = np.asarray(off, dtype=np.float64)
        if off_np.shape != weights_np.shape:
            raise ValueError(
                f"feature_weight_offset length {off_np.shape} must match "
                f"d={weights_np.shape}"
            )
        weights_np = weights_np + off_np
        if weights_end_np is not None:
            weights_end_np = weights_end_np + off_np
    elif isinstance(off, (int, float)) and float(off) != 0.0:
        weights_np = weights_np + float(off)
        if weights_end_np is not None:
            weights_end_np = weights_end_np + float(off)
    weights = torch.from_numpy(weights_np).float().to(device)
    weights_end = (torch.from_numpy(weights_end_np).float().to(device)
                   if weights_end_np is not None else None)

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
    # Optional selector warm-start / hard-prune prior.
    if cfg.selector_prior_logits:
        prior = np.asarray(cfg.selector_prior_logits, dtype=np.float32)
        if prior.shape[0] != X_train.shape[1]:
            raise ValueError(
                f"selector_prior_logits length {prior.shape[0]} must match "
                f"d={X_train.shape[1]}"
            )
        model.set_selector_prior(torch.from_numpy(prior))

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

        # ---- Optional: multi-sparsity training loss ----
        # When cfg.multi_k_targets is non-empty, sum prediction losses across
        # several top-k extractions from the same soft mask. The predictor
        # learns to predict at every sparsity it'll be asked about.
        multi_k = tuple(int(k) for k in cfg.multi_k_targets) if cfg.multi_k_targets else ()

        def _apply_train_topk(mask_in, k_now):
            """Hard top-k via straight-through, optionally dilated to patches."""
            topk_vals, topk_idx = torch.topk(mask_in, k=k_now, dim=1)
            hard_topk = torch.zeros_like(mask_in).scatter_(1, topk_idx, 1.0)
            if cfg.train_time_select_patch:
                hw = int(cfg.train_time_img_hw) \
                    or int(round(mask_in.shape[1] ** 0.5))
                hard_topk = _patch_dilate(hard_topk, img_hw=hw,
                                          kernel=int(cfg.train_time_patch_kernel))
            # Straight-through: forward = hard, backward = soft.
            return mask_in + (hard_topk - mask_in).detach()

        if multi_k:
            soft_for_topk = out["mask"]
            d_total = soft_for_topk.shape[1]
            losses_list = []
            for k_t in multi_k:
                if k_t <= 0 or k_t >= d_total:
                    mk = soft_for_topk
                else:
                    mk = _apply_train_topk(soft_for_topk, int(k_t))
                logits_k = model.predictor(x * mk)
                losses_list.append(
                    _classification_loss(logits_k, y, output_dim,
                                         label_smoothing=cfg.label_smoothing)
                )
            loss_pred = sum(losses_list) / len(losses_list)
        else:
            loss_pred = _classification_loss(out["logits"], y, output_dim,
                                             label_smoothing=cfg.label_smoothing)

        # ---- Optional: auxiliary deterministic-soft-mask prediction loss ----
        # Adds α·CE(predictor(x·sigmoid(μ_mean)), y). Gives the selector
        # direct prediction-gradient on every gate (essential for features
        # where x_i ≈ 0 — e.g. MNIST border pixels — that would otherwise
        # receive exactly zero gradient through the standard pipeline).
        if cfg.aux_soft_pred_weight and float(cfg.aux_soft_pred_weight) > 0:
            soft_det = model.gates_from_logits(
                out["mus"], stochastic=False,
                gate_type=cfg.gate_type, hc_beta=cfg.hc_beta,
            ).mean(dim=0)  # [B, d]
            aux_logits = model.predictor(x * soft_det)
            aux_loss = _classification_loss(
                aux_logits, y, output_dim,
                label_smoothing=cfg.label_smoothing,
            )
            loss_pred = loss_pred + float(cfg.aux_soft_pred_weight) * aux_loss

        sw = (bootstrap_weights(x.shape[0], cfg.num_selectors, device=device)
              if cfg.bootstrap else None)

        # γ-annealing: interpolate ω linearly over the post-warmup window.
        if anneal_active and cfg.max_iters > cfg.warmup_iters:
            anneal_lambda = max(0.0, min(
                1.0, (iter_idx - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
            ))
            reg_weights = weights + anneal_lambda * (weights_end - weights)
        else:
            reg_weights = weights
        loss_reg = gate_regularizer(
            mus=out["mus"], weights=reg_weights, sigma=model.sigma, sample_weights=sw,
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
    # ---- OPTIONAL EXTENSIONS (all default off ⇒ original behaviour) ----
    top_k: int | None = None,
    select_patch: bool = False,
    img_hw: int | None = None,
    patch_kernel: int = 3,
    nms_radius: int = 0,
    prior_score: np.ndarray | None = None,
    prior_blend: float = 0.5,
) -> dict:
    """Run inference and return predictions + per-sample binary masks.

    Default behaviour (no extension args): the predictor sees the
    majority-vote / soft mask from VIBMask's K selectors.

    Optional extension args (used by `examples/mnist/`):
      - `top_k`: keep only the K highest-scoring features per instance.
      - `select_patch`, `img_hw`, `patch_kernel`: after top-k, dilate each
        pick into a kernel×kernel 2D neighborhood (SUWR-style patches).
      - `nms_radius`: pick top-1, suppress a (2r+1)² 2D neighborhood,
        repeat — forces spatial diversity in selected pixels.
      - `prior_score`, `prior_blend`: blend a per-feature prior
        (e.g. Lasso importance) into the top-k ranking:
        `ranking = blend·prior + (1-blend)·soft_mask`.

    Returns a dict with:
      - 'logits':     [N, output_dim] numpy
      - 'preds':      [N] integer predictions
      - 'mask':       [N, d] binary mask (majority vote when hard=True)
      - 'soft_mask':  [N, d] continuous M̃ (no noise)
      - 'hard_mask':  [N, d] strict majority-vote binary mask
        (or per-instance top-K mask when `top_k` is set)
    """
    device = torch.device(device)
    model = model.to(device).eval()
    n = X.shape[0]
    out_logits, out_mask, out_soft, out_hard = [], [], [], []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.from_numpy(X[start:end]).float().to(device)
        out = model.predict(x, hard=hard)
        if top_k is not None and top_k > 0 and top_k < x.shape[1]:
            # Build a per-instance top-K mask, optionally dilated to patches.
            soft = out["soft_mask"]                                 # [B, d]
            hw = int(img_hw) if img_hw else int(round(soft.shape[-1] ** 0.5))
            if prior_score is not None:
                prior_t = torch.as_tensor(prior_score, dtype=soft.dtype,
                                          device=soft.device)
                if prior_t.shape != (soft.shape[-1],):
                    raise ValueError(
                        f"prior_score shape {tuple(prior_t.shape)} must be (d,)"
                    )
                ranking = (float(prior_blend) * prior_t.view(1, -1)
                           + (1.0 - float(prior_blend)) * soft)
            else:
                ranking = soft
            if int(nms_radius) > 0:
                topk_mask = _topk_nms(ranking, k=int(top_k), img_hw=hw,
                                      radius=int(nms_radius))
            else:
                _, topk_idx = torch.topk(ranking, k=int(top_k), dim=1)
                topk_mask = torch.zeros_like(soft).scatter_(1, topk_idx, 1.0)
            if select_patch:
                topk_mask = _patch_dilate(topk_mask, img_hw=hw,
                                          kernel=int(patch_kernel))
            logits = model.predictor(x * topk_mask)
            out_logits.append(logits.cpu().numpy())
            out_mask.append(topk_mask.cpu().numpy())
            out_soft.append(soft.cpu().numpy())
            out_hard.append(topk_mask.cpu().numpy())
        else:
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
