"""Linear policy + Welford-style state normalizer used by ARS-V2.

Two roles, one module:

1. :class:`LinearPolicy` carries the *current* mean policy
   :math:`\\theta = (W, b)` (no per-env perturbation) and serves as the
   exportable ONNX artifact.
2. :class:`ArsActorAdapter` wraps a :class:`LinearPolicy` with the
   :class:`RunningMeanStd` so it satisfies the ``actor(obs, stochastic_output=...)``
   contract expected by :func:`otus_rl_project.eval.run_locomotion_eval_rollout`.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

EPS_NORM = 1e-8

ACTOR_OBS_GROUP = "actor"


class RunningMeanStd:
  """Welford incremental mean / variance over a flat ``(N, D)`` tensor stream.

  Shared across train + eval so the deterministic eval rollout sees the same
  observation distribution as training.
  """

  def __init__(self, dim: int, device: str | torch.device) -> None:
    self.mean = torch.zeros(dim, device=device, dtype=torch.float32)
    self.var = torch.ones(dim, device=device, dtype=torch.float32)
    self.count = 1e-4

  def update(self, x: torch.Tensor) -> None:
    """Merge a ``(N, dim)`` batch into the running statistics (Chan et al., 1979)."""
    if x.numel() == 0:
      return
    flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    bm = flat.mean(dim=0)
    bv = flat.var(dim=0, unbiased=False)
    bn = float(flat.shape[0])
    delta = bm - self.mean
    tot = self.count + bn
    self.mean = self.mean + delta * (bn / tot)
    m_a = self.var * self.count
    m_b = bv * bn
    self.var = (m_a + m_b + (delta * delta) * (self.count * bn / tot)) / tot
    self.count = tot

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    """Affine ``(x - mean) / sqrt(var + eps)`` cast to ``x.dtype``."""
    std = torch.sqrt(self.var + EPS_NORM)
    return ((x.to(torch.float32) - self.mean) / std).to(x.dtype)

  def state_dict(self) -> dict[str, torch.Tensor | float]:  # noqa: D102
    return {"mean": self.mean.detach().cpu(), "var": self.var.detach().cpu(), "count": self.count}

  def load_state_dict(self, sd: dict, device: str | torch.device) -> None:  # noqa: D102
    self.mean = sd["mean"].to(device).to(torch.float32)
    self.var = sd["var"].to(device).to(torch.float32)
    self.count = float(sd["count"])


class LinearPolicy(nn.Module):
  """Affine map :math:`a = W \\cdot x_{norm} + b`, ``(obs_dim,) -> (action_dim,)``.

  Single set of parameters (no batched perturbations) — used as the *mean*
  policy and as the ONNX export module.
  """

  def __init__(self, obs_dim: int, action_dim: int) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.linear = nn.Linear(obs_dim, action_dim, bias=True)
    nn.init.zeros_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
    return self.linear(x)


class ArsActorAdapter:
  """Wrap :class:`LinearPolicy` + :class:`RunningMeanStd` to look like an rsl-rl actor.

  Matches the call signature used by
  :func:`otus_rl_project.eval.run_locomotion_eval_rollout`::

    action = actor(obs, stochastic_output=False)

  ARS is deterministic, so ``stochastic_output`` is ignored.
  """

  def __init__(
    self,
    policy: LinearPolicy,
    obs_norm: RunningMeanStd,
    *,
    clip_actions: float | None = None,
  ) -> None:
    self.policy = policy
    self.obs_norm = obs_norm
    self.clip_actions = clip_actions

  def eval(self) -> "ArsActorAdapter":  # noqa: D102
    self.policy.eval()
    return self

  def __call__(
    self, obs: TensorDict | dict | torch.Tensor, stochastic_output: bool = False
  ) -> torch.Tensor:
    """Extract the actor obs slice, normalize, apply linear map, optionally clip."""
    del stochastic_output
    if isinstance(obs, (TensorDict, dict)):
      x = obs[ACTOR_OBS_GROUP]
      assert isinstance(x, torch.Tensor)
    else:
      x = obs
    x_norm = self.obs_norm.normalize(x)
    a = self.policy(x_norm)
    if self.clip_actions is not None:
      a = a.clamp(-self.clip_actions, self.clip_actions)
    return a
