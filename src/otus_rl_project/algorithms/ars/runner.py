"""Augmented Random Search (V2-t) training loop with the shared eval-callback.

Vectorization layout: ``num_envs = N_dir × 2 × M_env_per_dir``. Each env carries its
own perturbed policy ``(W ± σ·δ_d, b ± σ·δ_d^b)``, so a single batched ``RslRlVecEnvWrapper.step``
collects rewards for all directions at once.

For algorithm details see Mania et al., 2018, *Simple Random Search of Static Linear
Policies is Competitive for Reinforcement Learning* (Algorithm 2 / V2-t).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from otus_rl_project.algorithms.ars.policy import (
  ACTOR_OBS_GROUP,
  ArsActorAdapter,
  LinearPolicy,
  RunningMeanStd,
)
from otus_rl_project.eval.locomotion_eval import (
  EVAL_PREFIX,
  LocomotionEvalPerturbation,
  log_eval_metrics,
  run_locomotion_eval_rollout,
)
from otus_rl_project.eval.perturbations import clear_pelvis_push

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

EPS_RSTD = 1e-6


@dataclass
class ArsRunnerCfg:
  """Hyperparameters + eval cadence for :class:`ArsRunner`.

  ``num_envs`` must equal ``num_directions × 2 × envs_per_direction`` so the
  vec env partitions cleanly into (direction × ± mirror × env replicas).
  """

  num_directions: int = 64
  """Number of i.i.d. perturbation directions per iteration (``N`` in the paper)."""
  top_directions: int = 32
  """Top-:math:`b` filter on ``max(r+, r-)``. Must be ≤ ``num_directions``."""
  rollout_steps: int = 200
  """Steps per rollout. Each iteration runs one rollout for all ``num_envs``."""
  step_size: float = 0.02
  """ARS learning rate :math:`\\alpha`."""
  noise_std: float = 0.03
  """Perturbation magnitude :math:`\\sigma` (Gaussian on θ)."""
  clip_actions: float = 4.0
  """Symmetric ``|a| ≤ clip`` so pre-trained-σ noise can't tear the sim apart."""
  reward_to_go_clip: float | None = None
  """Optional per-step reward clip (``None`` disables). Helps if a reward term spikes."""

  eval_interval: int = 25
  """0 disables. One tick runs nominal + obs-noise + 5 N + 15 N (sequential)."""
  eval_num_envs: int = 256
  eval_num_steps: int = 96
  eval_obs_noise_std: float = 0.05
  eval_threshold_mps: float = 0.15
  """Sample-efficiency threshold for ``Eval/env_steps_to_threshold``."""

  save_interval: int = 50
  log_interval: int = 1
  seed: int = 42
  experiment_name: str = "g1_walk_compare_ars"
  run_name: str = ""

  def __post_init__(self) -> None:
    if self.top_directions > self.num_directions:
      raise ValueError(
        f"top_directions={self.top_directions} > num_directions={self.num_directions}"
      )

  def expected_num_envs(self, envs_per_direction: int) -> int:
    return self.num_directions * 2 * envs_per_direction


@dataclass
class _IterStats:
  iteration: int = 0
  total_env_steps: int = 0
  best_reward: float = -float("inf")
  history: list[float] = field(default_factory=list)


class ArsRunner:
  """ARS-V2 training driver. Keeps API symmetric with the PPO compare runner.

  Public surface:

  - :meth:`learn` — train for ``num_iterations`` updates and return final reward.
  - :meth:`save` / :meth:`load` — pickle the policy + obs normalizer (``.pt``).
  - :meth:`export_policy_to_onnx` — emit a single ``policy.onnx`` (linear + normalizer baked-in).
  """

  def __init__(
    self,
    train_env: RslRlVecEnvWrapper,
    eval_env_factory,
    cfg: ArsRunnerCfg,
    *,
    log_dir: str,
    device: str = "cuda:0",
  ) -> None:
    self.cfg = cfg
    self.train_env = train_env
    self._eval_env_factory = eval_env_factory
    self._eval_vec: RslRlVecEnvWrapper | None = None
    self.device = device
    self.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    self.num_envs = train_env.num_envs
    self.action_dim = train_env.num_actions
    obs_dict = train_env.get_observations()
    actor_obs = obs_dict[ACTOR_OBS_GROUP] if isinstance(obs_dict, (dict, TensorDict)) else obs_dict
    assert isinstance(actor_obs, torch.Tensor)
    self.obs_dim = int(actor_obs.shape[-1])

    if cfg.expected_num_envs(self._envs_per_direction()) != self.num_envs:
      raise ValueError(
        f"num_envs={self.num_envs} must equal "
        f"num_directions ({cfg.num_directions}) × 2 × envs_per_direction "
        f"({self._envs_per_direction()}) = "
        f"{cfg.expected_num_envs(self._envs_per_direction())}."
      )

    torch.manual_seed(cfg.seed)
    self.policy = LinearPolicy(self.obs_dim, self.action_dim).to(device)
    self.obs_norm = RunningMeanStd(self.obs_dim, device)
    self.writer: SummaryWriter | None = SummaryWriter(log_dir=log_dir)
    self.stats = _IterStats()
    self._first_threshold_logged = False
    self._wall_t0: float | None = None

  def _envs_per_direction(self) -> int:
    return self.num_envs // (self.cfg.num_directions * 2)

  # ------------------------------------------------------------------ rollout

  def _build_perturbed_params(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``N_dir`` deltas, broadcast to per-env (W, b) tensors with ± mirrors."""
    g = torch.Generator(device=self.device).manual_seed(
      self.cfg.seed + self.stats.iteration + 1
    )
    n = self.cfg.num_directions
    m = self._envs_per_direction()
    sigma = self.cfg.noise_std
    dW = torch.randn(
      n, self.action_dim, self.obs_dim, device=self.device, generator=g
    )
    db = torch.randn(n, self.action_dim, device=self.device, generator=g)

    base_W = self.policy.linear.weight.detach()
    base_b = self.policy.linear.bias.detach()
    sign = torch.tensor([1.0, -1.0], device=self.device).view(1, 2, 1, 1)
    W_dirs = base_W.view(1, 1, self.action_dim, self.obs_dim) + sigma * sign * dW.view(
      n, 1, self.action_dim, self.obs_dim
    )
    b_dirs = base_b.view(1, 1, self.action_dim) + sigma * sign.view(1, 2, 1) * db.view(
      n, 1, self.action_dim
    )
    W_env = W_dirs.unsqueeze(2).expand(n, 2, m, self.action_dim, self.obs_dim).reshape(
      self.num_envs, self.action_dim, self.obs_dim
    )
    b_env = b_dirs.unsqueeze(2).expand(n, 2, m, self.action_dim).reshape(
      self.num_envs, self.action_dim
    )
    return W_env, b_env, dW, db

  def _rollout_collect_rewards(
    self, W_env: torch.Tensor, b_env: torch.Tensor
  ) -> torch.Tensor:
    """One rollout for all envs in parallel; returns ``(num_envs,)`` cumulative reward."""
    obs_td, _ = self.train_env.reset()
    obs = obs_td[ACTOR_OBS_GROUP].to(self.device)
    rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    alive = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
    clip = self.cfg.clip_actions
    rew_clip = self.cfg.reward_to_go_clip

    for _ in range(self.cfg.rollout_steps):
      self.obs_norm.update(obs)
      x = self.obs_norm.normalize(obs).unsqueeze(-1)
      action = torch.bmm(W_env, x).squeeze(-1) + b_env
      action = action.clamp(-clip, clip)

      next_td, rew, dones, _ = self.train_env.step(action)
      r = rew.float()
      if rew_clip is not None:
        r = r.clamp(-rew_clip, rew_clip)
      rewards = rewards + alive * r
      alive = alive * (1.0 - dones.float())
      obs = next_td[ACTOR_OBS_GROUP].to(self.device)

    return rewards

  # ------------------------------------------------------------------ update

  def _ars_update(
    self,
    rewards: torch.Tensor,
    dW: torch.Tensor,
    db: torch.Tensor,
  ) -> tuple[float, float]:
    """ARS-V2-t: top-:math:`b` filter, divide by σ_R of the kept ± rewards."""
    n = self.cfg.num_directions
    m = self._envs_per_direction()
    per_dir = rewards.view(n, 2, m).mean(dim=-1)
    r_plus = per_dir[:, 0]
    r_minus = per_dir[:, 1]
    score = torch.maximum(r_plus, r_minus)

    b_top = self.cfg.top_directions
    top_idx = torch.topk(score, b_top, largest=True).indices
    rp = r_plus[top_idx]
    rm = r_minus[top_idx]
    sigma_R = torch.std(torch.cat([rp, rm]), unbiased=False)
    diff = (rp - rm).view(b_top, 1, 1)

    update_W = (diff * dW[top_idx]).sum(dim=0) / (b_top * (sigma_R + EPS_RSTD))
    update_b = (diff.view(b_top, 1) * db[top_idx]).sum(dim=0) / (b_top * (sigma_R + EPS_RSTD))

    with torch.no_grad():
      self.policy.linear.weight.add_(self.cfg.step_size * update_W)
      self.policy.linear.bias.add_(self.cfg.step_size * update_b)

    mean_rew = float(per_dir.mean().item())
    return mean_rew, float(score.max().item())

  # ------------------------------------------------------------------ eval

  def _ensure_eval_vec(self) -> RslRlVecEnvWrapper:
    if self._eval_vec is None:
      self._eval_vec = self._eval_env_factory(self.cfg.eval_num_envs)
    return self._eval_vec

  def _run_eval_phase(self, it: int) -> None:
    if self.cfg.eval_interval <= 0 or self.writer is None:
      return
    if it % self.cfg.eval_interval != 0:
      return
    if self._wall_t0 is None:
      self._wall_t0 = time.monotonic()

    ev = self._ensure_eval_vec()
    actor = ArsActorAdapter(self.policy, self.obs_norm, clip_actions=self.cfg.clip_actions)

    seq = [
      (LocomotionEvalPerturbation.NONE, "", {}),
      (LocomotionEvalPerturbation.OBS_NOISE, "obs_noise", {"obs_noise_std": self.cfg.eval_obs_noise_std}),
      (LocomotionEvalPerturbation.PUSH_5N, "push_5N", {}),
      (LocomotionEvalPerturbation.PUSH_15N, "push_15N", {}),
    ]
    vel_key = f"{EVAL_PREFIX}/velocity_tracking_error_xy"
    for pert, suffix, extra in seq:
      m = run_locomotion_eval_rollout(
        actor,
        ev,
        self.device,
        num_steps=self.cfg.eval_num_steps,
        perturbation=pert,
        **extra,
      )
      log_eval_metrics(self.writer, m, it, key_suffix=suffix)
      v_err = m.get(vel_key, 0.0)
      if pert is LocomotionEvalPerturbation.OBS_NOISE:
        self.writer.add_scalar(f"{EVAL_PREFIX}/vel_error_obs_noise", v_err, it)
      elif pert is LocomotionEvalPerturbation.PUSH_5N:
        self.writer.add_scalar(f"{EVAL_PREFIX}/vel_error_push_5N", v_err, it)
      elif pert is LocomotionEvalPerturbation.PUSH_15N:
        self.writer.add_scalar(f"{EVAL_PREFIX}/vel_error_push_15N", v_err, it)

      if not self._first_threshold_logged and pert is LocomotionEvalPerturbation.NONE:
        if m.get(vel_key, 1e6) < self.cfg.eval_threshold_mps:
          self._first_threshold_logged = True
          self.writer.add_scalar(
            f"{EVAL_PREFIX}/env_steps_to_threshold",
            int(self.stats.total_env_steps),
            it,
          )
          self.writer.add_scalar(
            f"{EVAL_PREFIX}/wall_clock_to_threshold",
            time.monotonic() - float(self._wall_t0 or time.monotonic()),
            it,
          )

  # ------------------------------------------------------------------ save / load

  def save(self, path: str) -> None:  # noqa: D102
    torch.save(
      {
        "policy_state_dict": self.policy.state_dict(),
        "obs_norm": self.obs_norm.state_dict(),
        "iteration": self.stats.iteration,
        "total_env_steps": self.stats.total_env_steps,
        "obs_dim": self.obs_dim,
        "action_dim": self.action_dim,
      },
      path,
    )

  def load(self, path: str) -> None:  # noqa: D102
    sd = torch.load(path, map_location=self.device, weights_only=False)
    self.policy.load_state_dict(sd["policy_state_dict"])
    self.obs_norm.load_state_dict(sd["obs_norm"], device=self.device)
    self.stats.iteration = int(sd.get("iteration", 0))
    self.stats.total_env_steps = int(sd.get("total_env_steps", 0))

  def export_policy_to_onnx(self, dir_path: str, filename: str = "policy.onnx") -> str:
    """Emit a single ``(obs_dim,) -> (action_dim,)`` ONNX with normalizer fused."""
    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, filename)

    mean = self.obs_norm.mean.detach()
    std = torch.sqrt(self.obs_norm.var.detach() + 1e-8)
    W = self.policy.linear.weight.detach()
    b_old = self.policy.linear.bias.detach()
    fused = LinearPolicy(self.obs_dim, self.action_dim).to("cpu").eval()
    with torch.no_grad():
      fused.linear.weight.copy_((W / std).cpu())
      fused.linear.bias.copy_((b_old - (W @ (mean / std))).cpu())

    dummy = torch.zeros(1, self.obs_dim)
    torch.onnx.export(
      fused,
      (dummy,),
      out_path,
      input_names=["obs"],
      output_names=["actions"],
      opset_version=18,
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      dynamo=False,
    )
    return out_path

  # ------------------------------------------------------------------ learn

  def learn(self, num_iterations: int) -> float:  # noqa: D102
    self._wall_t0 = time.monotonic()
    last_mean = -float("inf")
    for it in range(1, num_iterations + 1):
      self.stats.iteration = it
      W_env, b_env, dW, db = self._build_perturbed_params()
      rewards = self._rollout_collect_rewards(W_env, b_env)
      mean_rew, max_rew = self._ars_update(rewards, dW, db)
      self.stats.total_env_steps += int(self.num_envs * self.cfg.rollout_steps)
      self.stats.history.append(mean_rew)
      self.stats.best_reward = max(self.stats.best_reward, max_rew)
      last_mean = mean_rew

      if self.writer is not None and (it % self.cfg.log_interval == 0):
        self.writer.add_scalar("Train/mean_reward_per_direction", mean_rew, it)
        self.writer.add_scalar("Train/max_reward_per_direction", max_rew, it)
        self.writer.add_scalar("Train/total_env_steps", self.stats.total_env_steps, it)
        self.writer.add_scalar(
          "Train/wall_clock_seconds", time.monotonic() - (self._wall_t0 or time.monotonic()), it
        )

      self._run_eval_phase(it)

      if self.cfg.save_interval > 0 and (it % self.cfg.save_interval == 0):
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      print(
        f"[ARS] it={it:5d}  mean_R={mean_rew:+.3f}  max_R={max_rew:+.3f}  "
        f"steps={self.stats.total_env_steps}",
        flush=True,
      )

    final_pt = os.path.join(self.log_dir, f"model_{self.stats.iteration}.pt")
    self.save(final_pt)
    try:
      self.export_policy_to_onnx(self.log_dir, "policy.onnx")
    except Exception as e:
      print(f"[ARS] ONNX export failed (training continues): {e}", flush=True)

    if self._eval_vec is not None:
      try:
        with torch.inference_mode():
          clear_pelvis_push(self._eval_vec.unwrapped)
      except Exception:  # noqa: S110
        pass
      self._eval_vec.close()
      self._eval_vec = None

    if self.writer is not None:
      self.writer.flush()
      self.writer.close()

    return last_mean


def build_eval_env_factory(train_env: RslRlVecEnvWrapper):
  """Closure that builds a smaller eval vec env (frozen cfg, fresh sim)."""
  from dataclasses import replace

  def factory(num_envs: int) -> RslRlVecEnvWrapper:
    src = train_env.unwrapped
    n = min(int(num_envs), int(src.num_envs))
    new_cfg = replace(src.cfg, scene=replace(src.cfg.scene, num_envs=n))
    dev = str(src.device) if not isinstance(src.device, str) else src.device
    menv = ManagerBasedRlEnv(cfg=new_cfg, device=dev, render_mode=None)
    return RslRlVecEnvWrapper(menv, clip_actions=getattr(train_env, "clip_actions", None))

  return factory
