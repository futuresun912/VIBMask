"""VIBMask model: K local selectors + shared MLP predictor.

Implements the architecture in Eq. (4) of the paper:
    M̃_k = clip(μ_k + σ·ε, 0, 1),  μ_k = g_{ψ_k}(X),  ε ~ N(0, I)
    M̃   = (1/K) Σ_k M̃_k
    ŷ   = f_θ(X ⊙ M̃)

At inference (ε = 0) each M̃_k is thresholded at 0.5 and a binary mask is
formed by majority vote across selectors (paper §3.4).
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


def _xavier_init_(module: nn.Module) -> None:
    """Xavier uniform on every nn.Linear (paper Algorithm 1 line 1)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module],
    layer_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Standard MLP with Xavier-uniform init. Optional LayerNorm and dropout
    after every hidden activation (not after the output projection).
    """
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    net = nn.Sequential(*layers)
    net.apply(_xavier_init_)
    return net


class Selector(nn.Module):
    """Single instance-wise selector g_ψ: R^d → R^d producing logits μ.

    The output μ is unbounded; the relaxed Bernoulli gate is formed
    downstream as clip(μ + σ·ε, 0, 1) (gaussian_clip) or via the
    hard-concrete distribution. Following STG / LSPIN we initialise μ
    near 0.5 by biasing the final layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (100, 100, 100),
        activation: type[nn.Module] = nn.LeakyReLU,
        init_bias: float = 0.5,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = _mlp(input_dim, list(hidden_dims), input_dim, activation,
                        layer_norm=layer_norm)
        # Bias the final layer so μ ≈ init_bias at init -> gates start near 0.5.
        with torch.no_grad():
            self.net[-1].bias.fill_(init_bias)
            self.net[-1].weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """Shared MLP predictor f_θ. Returns raw logits (no sigmoid/softmax)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (100, 100, 10),
        activation: type[nn.Module] = nn.LeakyReLU,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = _mlp(input_dim, list(hidden_dims), output_dim, activation,
                        layer_norm=layer_norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VIBMask(nn.Module):
    """VIBMask: K selectors + shared predictor (paper Algorithm 1).

    Args:
        input_dim: feature dimension d.
        output_dim: number of classes (or 1 for binary BCE).
        num_selectors: K, the number of selectors.
        sigma: noise scale σ for the relaxed Bernoulli gate.
        sel_hidden: hidden sizes for each selector MLP.
        pred_hidden: hidden sizes for the predictor MLP.
        train_sigma: if True, σ is a learnable scalar (softplus-clipped > 0).
        layer_norm: insert LayerNorm after each Linear hidden activation.
        activation: hidden activation, defaults to LeakyReLU.
        gate_type: "gaussian_clip" (paper default) or "hard_concrete".
        hc_beta: hard-concrete temperature β (smaller = sharper).
        pred_dropout: dropout after each predictor hidden activation.

    `forward` returns a dict with logits + per-selector μ for the regulariser.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_selectors: int = 3,
        sigma: float = 0.5,
        sel_hidden: Sequence[int] = (100, 100, 100),
        pred_hidden: Sequence[int] = (100, 100, 10),
        train_sigma: bool = False,
        layer_norm: bool = False,
        activation: type[nn.Module] = nn.LeakyReLU,
        gate_type: str = "gaussian_clip",
        hc_beta: float = 0.5,
        pred_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_selectors < 1:
            raise ValueError("num_selectors must be >= 1")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_selectors = num_selectors
        self.train_sigma = train_sigma
        self.gate_type = gate_type
        self.hc_beta = float(hc_beta)

        if train_sigma:
            self._raw_sigma = nn.Parameter(torch.tensor(float(sigma)))
        else:
            self.register_buffer("_raw_sigma", torch.tensor(float(sigma)))

        self.selectors = nn.ModuleList(
            [
                Selector(input_dim, sel_hidden, activation=activation, layer_norm=layer_norm)
                for _ in range(num_selectors)
            ]
        )
        self.predictor = Predictor(
            input_dim, output_dim, pred_hidden, activation=activation,
            layer_norm=layer_norm, dropout=pred_dropout,
        )
        # Optional warm-up / hard-prune prior added as a constant bias to
        # every selector's μ output. Default zeros = no prior (original
        # behaviour). Use `set_selector_prior()` to inject a length-d
        # vector (e.g. variance log-prior or Lasso scores) for the
        # MNIST-style extensions. A very negative value (~-1000) pins the
        # corresponding gate to ≈0 — used by V16-A hard-prune.
        self.register_buffer(
            "_selector_prior",
            torch.zeros(input_dim, dtype=torch.float32),
        )

    @property
    def sigma(self) -> torch.Tensor:
        if self.train_sigma:
            return torch.nn.functional.softplus(self._raw_sigma) + 1e-3
        return self._raw_sigma

    def set_selector_prior(self, prior: torch.Tensor) -> None:
        """Inject a length-d bias added to every selector's μ output.

        See `TrainConfig.selector_prior_logits`. Default value is all
        zeros, which makes the prior a no-op (original VIBMask behaviour).
        """
        if prior.shape != self._selector_prior.shape:
            raise ValueError(
                f"selector prior shape {tuple(prior.shape)} != input_dim "
                f"{tuple(self._selector_prior.shape)}"
            )
        self._selector_prior = prior.to(self._selector_prior.device).float()

    def selector_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Stack of per-selector logits μ. Shape: [K, B, d].

        Adds `_selector_prior` (default zeros — no-op) as a constant bias.
        """
        mus = torch.stack([s(x) for s in self.selectors], dim=0)
        return mus + self._selector_prior.view(1, 1, -1)

    def gates_from_logits(self, mus: torch.Tensor, stochastic: bool,
                          gate_type: str = "gaussian_clip",
                          hc_beta: float = 0.5) -> torch.Tensor:
        """Relaxed-Bernoulli reparameterisation of logits μ.

        - "gaussian_clip" (paper default): z = clip(μ + σ·ε, 0, 1).
        - "hard_concrete": Louizos 2018 hard-concrete distribution stretched
          to (γ=-0.1, ζ=1.1) so 0 and 1 receive positive probability mass.
        """
        if gate_type == "gaussian_clip":
            if stochastic:
                eps = torch.randn_like(mus)
                z = mus + self.sigma * eps
            else:
                z = mus
            return torch.clamp(z, 0.0, 1.0)
        elif gate_type == "hard_concrete":
            zeta, gamma_lim = 1.1, -0.1
            if stochastic:
                u = torch.rand_like(mus).clamp(1e-6, 1.0 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + mus) / hc_beta)
            else:
                s = torch.sigmoid(mus / hc_beta)
            s_bar = s * (zeta - gamma_lim) + gamma_lim
            return torch.clamp(s_bar, 0.0, 1.0)
        else:
            raise ValueError(f"unknown gate_type {gate_type!r}")

    def aggregate_mask(self, gates: torch.Tensor) -> torch.Tensor:
        """Average gates across selectors. gates: [K, B, d] -> [B, d]."""
        return gates.mean(dim=0)

    def majority_vote(self, gates: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Per-feature majority vote across K selectors.

        gates: [K, B, d]. Returns binary [B, d] tensor on the same device.
        Tie-break (count == K/2 when K is even): not selected, matching the
        strict ">K/2" reading of "majority" in the paper.
        """
        binary = (gates >= threshold).to(gates.dtype)
        votes = binary.sum(dim=0)
        return (votes > self.num_selectors / 2.0).to(gates.dtype)

    def forward(self, x: torch.Tensor) -> dict:
        """Training-mode forward pass.

        Returns dict with:
          - logits: [B, output_dim]
          - mus:    [K, B, d] selector logits (for regulariser)
          - gates:  [K, B, d] soft gates (stochastic when self.training)
          - mask:   [B, d] aggregated soft mask M̃
        """
        mus = self.selector_logits(x)
        gates = self.gates_from_logits(
            mus, stochastic=self.training,
            gate_type=self.gate_type, hc_beta=self.hc_beta,
        )
        mask = self.aggregate_mask(gates)
        logits = self.predictor(x * mask)
        return {"logits": logits, "mus": mus, "gates": gates, "mask": mask}

    @torch.no_grad()
    def predict(self, x: torch.Tensor, hard: bool = True) -> dict:
        """Deterministic inference. Returns both hard (majority-vote) and
        soft (mean) masks so eval can pick its preferred interpretation.
        """
        was_training = self.training
        self.eval()
        try:
            mus = self.selector_logits(x)
            gates = self.gates_from_logits(
                mus, stochastic=False,
                gate_type=self.gate_type, hc_beta=self.hc_beta,
            )
            soft_mask = self.aggregate_mask(gates)
            hard_mask = self.majority_vote(gates)
            used_mask = hard_mask if hard else soft_mask
            logits = self.predictor(x * used_mask)
        finally:
            self.train(was_training)
        return {
            "logits": logits, "mus": mus, "gates": gates,
            "soft_mask": soft_mask, "hard_mask": hard_mask, "mask": used_mask,
        }
