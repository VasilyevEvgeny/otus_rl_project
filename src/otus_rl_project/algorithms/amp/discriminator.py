"""LSGAN-style AMP discriminator (Peng et al., 2021, *AMP: Adversarial Motion Priors*).

Inputs are concatenated state pairs ``(s, s')``; output is a single scalar.
We use LSGAN targets ``+1`` for expert and ``-1`` for policy and a one-sided
gradient penalty on expert samples to keep the discriminator well-conditioned.

Style reward (Peng et al., eq. 5)::

    r_style(s, s') = max(0, 1 - 0.25 * (D(s, s') - 1)^2)
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class AmpDiscriminator(nn.Module):
  """MLP scoring (s, s') ∈ ℝ^{2F}; LSGAN targets {+1=expert, -1=policy}."""

  def __init__(
    self,
    feature_dim: int,
    hidden_dims: Iterable[int] = (1024, 512),
    activation: type[nn.Module] = nn.ReLU,
  ) -> None:
    super().__init__()
    self.feature_dim = int(feature_dim)
    layers: list[nn.Module] = []
    prev = self.feature_dim * 2
    for h in hidden_dims:
      layers.append(nn.Linear(prev, h))
      layers.append(activation())
      prev = h
    layers.append(nn.Linear(prev, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:  # noqa: D102
    x = torch.cat([s, sp], dim=-1)
    return self.net(x).squeeze(-1)

  def compute_reward(self, s: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
    """Eq. 5 of Peng et al.: clamped quadratic on the LSGAN expert target."""
    with torch.no_grad():
      d = self.forward(s, sp)
      r = 1.0 - 0.25 * (d - 1.0).pow(2)
      return torch.clamp(r, min=0.0)

  def compute_grad_penalty(
    self, s_e: torch.Tensor, sp_e: torch.Tensor
  ) -> torch.Tensor:
    """One-sided WGAN-style ``E_expert[||∇D||²]`` (no |·|−1; just minimize norm)."""
    s = s_e.detach().requires_grad_(True)
    sp = sp_e.detach().requires_grad_(True)
    d = self.forward(s, sp)
    grads = torch.autograd.grad(
      outputs=d.sum(),
      inputs=[s, sp],
      create_graph=True,
      retain_graph=True,
    )
    return sum(g.pow(2).sum(dim=-1) for g in grads).mean()

  def compute_disc_loss(
    self,
    s_p: torch.Tensor,
    sp_p: torch.Tensor,
    s_e: torch.Tensor,
    sp_e: torch.Tensor,
    lambda_gp: float,
  ) -> dict[str, torch.Tensor]:
    """LSGAN objective + grad penalty; returns scalar tensors for logging."""
    d_p = self.forward(s_p, sp_p)
    d_e = self.forward(s_e, sp_e)
    loss_policy = (d_p + 1.0).pow(2).mean()
    loss_expert = (d_e - 1.0).pow(2).mean()
    gp = self.compute_grad_penalty(s_e, sp_e)
    total = 0.5 * (loss_policy + loss_expert) + lambda_gp * gp
    return {
      "disc_loss": total,
      "disc_loss_policy": loss_policy.detach(),
      "disc_loss_expert": loss_expert.detach(),
      "disc_grad_penalty": gp.detach(),
      "disc_d_policy_mean": d_p.detach().mean(),
      "disc_d_expert_mean": d_e.detach().mean(),
    }
