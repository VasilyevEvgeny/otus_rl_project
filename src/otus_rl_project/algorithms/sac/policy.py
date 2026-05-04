"""Networks + replay buffer for SAC (squashed Gaussian, twin critics, GPU ring buffer).

Mirrors the ARS module layout (``policy.py`` for nets, ``runner.py`` for the
training loop). The eval adapter (:class:`SacActorAdapter`) follows the same
``actor(obs, stochastic_output=False) -> action`` contract used by
:func:`otus_rl_project.eval.locomotion_eval.run_locomotion_eval_rollout`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from tensordict import TensorDict
from torch import nn

ACTOR_OBS_GROUP = "actor"
CRITIC_OBS_GROUP = "critic"

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  prev = in_dim
  for h in hidden:
    layers.append(nn.Linear(prev, h))
    layers.append(nn.ELU(inplace=False))
    prev = h
  layers.append(nn.Linear(prev, out_dim))
  return nn.Sequential(*layers)


def _select_obs_group(obs: TensorDict | dict | torch.Tensor, group: str) -> torch.Tensor:
  if isinstance(obs, (TensorDict, dict)):
    x = obs[group]
    assert isinstance(x, torch.Tensor)
    return x
  return obs


class GaussianActor(nn.Module):
  """Tanh-squashed diagonal-Gaussian policy.

  Forward returns ``(action, log_prob, tanh_pre_action)``. ``action`` is
  scaled by :attr:`action_scale` so the env sees actions roughly in
  ``[-action_scale, action_scale]`` (the tanh saturates near the boundary).

  ONNX export uses :meth:`forward_deterministic` (mean + tanh, no sampling),
  matching the deployment story used by :class:`otus_rl_project.deploy.export_onnx`.
  """

  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
    action_scale: float = 1.0,
  ) -> None:
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.action_dim = int(action_dim)
    self.action_scale = float(action_scale)
    self.backbone = _mlp(obs_dim, list(hidden_dims), int(hidden_dims[-1]))
    self.head_mean = nn.Linear(int(hidden_dims[-1]), action_dim)
    self.head_log_std = nn.Linear(int(hidden_dims[-1]), action_dim)

  def _features(self, obs: torch.Tensor) -> torch.Tensor:
    return self.backbone(obs)

  def forward(
    self, obs: torch.Tensor, *, deterministic: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (or take mean of) the squashed Gaussian; return ``(action, log_prob)``."""
    f = self._features(obs)
    mu = self.head_mean(f)
    if deterministic:
      a = torch.tanh(mu) * self.action_scale
      lp = torch.zeros(a.shape[:-1], device=a.device, dtype=a.dtype)
      return a, lp
    log_std = self.head_log_std(f).clamp(LOG_STD_MIN, LOG_STD_MAX)
    std = log_std.exp()
    eps = torch.randn_like(mu)
    u = mu + std * eps
    t = torch.tanh(u)
    a = t * self.action_scale
    # log_prob of squashed Gaussian: log N(u; mu, sigma^2) - sum_i log(1 - tanh(u_i)^2)
    # Numerically stable: log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u))
    log_prob_u = -0.5 * ((u - mu) / std).pow(2) - log_std - 0.5 * math.log(2.0 * math.pi)
    log_correction = 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))
    log_prob = (log_prob_u - log_correction).sum(dim=-1)
    log_prob = log_prob - self.action_dim * math.log(self.action_scale)
    return a, log_prob

  def forward_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
    """ONNX-export path: ``tanh(mu) * action_scale`` with no sampling."""
    f = self._features(obs)
    return torch.tanh(self.head_mean(f)) * self.action_scale


class TwinCritic(nn.Module):
  """Two MLP critics ``Q_i(s, a) -> ℝ`` sharing input shape; trained jointly via min-Q target."""

  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    *,
    hidden_dims: Sequence[int] = (256, 256, 256),
  ) -> None:
    super().__init__()
    self.q1 = _mlp(int(obs_dim) + int(action_dim), list(hidden_dims), 1)
    self.q2 = _mlp(int(obs_dim) + int(action_dim), list(hidden_dims), 1)

  def forward(
    self, obs: torch.Tensor, action: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([obs, action], dim=-1)
    return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
  """GPU-resident circular buffer over ``(s_actor, s_critic, a, r, s'_actor, s'_critic, d)``.

  Stores actor and critic obs separately so the SAC update path can feed each
  network the exact slice it expects (``ACTOR_OBS_GROUP`` for the policy,
  ``CRITIC_OBS_GROUP`` for the Q networks). When the env exposes only a single
  obs group, both pointers reference the same tensor and memory cost is unchanged.
  """

  def __init__(
    self,
    capacity: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    *,
    device: str | torch.device,
  ) -> None:
    self.capacity = int(capacity)
    self.device = device
    self.size = 0
    self.ptr = 0
    self.s_actor = torch.zeros((self.capacity, actor_obs_dim), device=device, dtype=torch.float32)
    self.s_critic = torch.zeros(
      (self.capacity, critic_obs_dim), device=device, dtype=torch.float32
    )
    self.a = torch.zeros((self.capacity, action_dim), device=device, dtype=torch.float32)
    self.r = torch.zeros((self.capacity,), device=device, dtype=torch.float32)
    self.sp_actor = torch.zeros((self.capacity, actor_obs_dim), device=device, dtype=torch.float32)
    self.sp_critic = torch.zeros(
      (self.capacity, critic_obs_dim), device=device, dtype=torch.float32
    )
    self.d = torch.zeros((self.capacity,), device=device, dtype=torch.float32)

  def push(
    self,
    s_actor: torch.Tensor,
    s_critic: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sp_actor: torch.Tensor,
    sp_critic: torch.Tensor,
    d: torch.Tensor,
  ) -> None:
    """Append a flat batch of transitions; wraps when capacity is hit."""
    n = int(s_actor.shape[0])
    if n == 0:
      return
    cap = self.capacity
    end = self.ptr + n
    if end <= cap:
      sl = slice(self.ptr, end)
      self.s_actor[sl].copy_(s_actor)
      self.s_critic[sl].copy_(s_critic)
      self.a[sl].copy_(a)
      self.r[sl].copy_(r)
      self.sp_actor[sl].copy_(sp_actor)
      self.sp_critic[sl].copy_(sp_critic)
      self.d[sl].copy_(d)
    else:
      first = cap - self.ptr
      sl1 = slice(self.ptr, cap)
      self.s_actor[sl1].copy_(s_actor[:first])
      self.s_critic[sl1].copy_(s_critic[:first])
      self.a[sl1].copy_(a[:first])
      self.r[sl1].copy_(r[:first])
      self.sp_actor[sl1].copy_(sp_actor[:first])
      self.sp_critic[sl1].copy_(sp_critic[:first])
      self.d[sl1].copy_(d[:first])
      rem = n - first
      sl2 = slice(0, rem)
      self.s_actor[sl2].copy_(s_actor[first:])
      self.s_critic[sl2].copy_(s_critic[first:])
      self.a[sl2].copy_(a[first:])
      self.r[sl2].copy_(r[first:])
      self.sp_actor[sl2].copy_(sp_actor[first:])
      self.sp_critic[sl2].copy_(sp_critic[first:])
      self.d[sl2].copy_(d[first:])
    self.size = min(self.size + n, cap)
    self.ptr = end % cap

  def sample(
    self, batch_size: int
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Uniform random sampling over the populated prefix."""
    n = min(int(batch_size), self.size)
    idx = torch.randint(0, self.size, (n,), device=self.s_actor.device)
    return (
      self.s_actor[idx],
      self.s_critic[idx],
      self.a[idx],
      self.r[idx],
      self.sp_actor[idx],
      self.sp_critic[idx],
      self.d[idx],
    )


class SacActorAdapter:
  """Wraps :class:`GaussianActor` so it satisfies the eval-callback contract.

  ARS deterministic; PPO/AMP stochastic-by-default but eval-rollouts request
  ``stochastic_output=False`` for reproducibility — SAC follows the same pattern
  via ``forward(deterministic=True)`` (mean of the squashed Gaussian).
  """

  def __init__(self, policy: GaussianActor) -> None:
    self.policy = policy

  def eval(self) -> "SacActorAdapter":  # noqa: D102
    self.policy.eval()
    return self

  def __call__(
    self, obs: TensorDict | dict | torch.Tensor, stochastic_output: bool = False
  ) -> torch.Tensor:
    x = _select_obs_group(obs, ACTOR_OBS_GROUP)
    a, _ = self.policy(x, deterministic=not stochastic_output)
    return a
