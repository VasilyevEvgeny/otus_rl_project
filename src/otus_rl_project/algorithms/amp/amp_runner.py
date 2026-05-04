"""AMP-augmented PPO runner for the locomotion compare task.

Sits on top of :class:`~otus_rl_project.eval.compare_runner.LocomotionCompareVelocityRunner`
so it inherits the shared :data:`Eval/*` callback. The AMP-specific pieces are:

1. A monkey-patched ``env.step`` that captures the AMP feature vector
   ``f(s) = [proj_grav_b, lin_vel_b, ang_vel_b, joint_pos]`` (38-D) right before
   and right after each sim step, blends an AMP style reward into the env
   reward, and pushes the (s, s') pair (excluding done-spanning ones) into a
   replay buffer.
2. A discriminator + Adam optimizer trained once per PPO iteration on a fresh
   batch of policy pairs vs expert pairs.

The reward blend is ``r = (1-w)·r_task + w·r_amp_scale·r_style`` with
``w = amp_reward_weight`` and ``r_amp_scale = amp_reward_scale``. When
``amp_replace_style_terms=True``, the env's hand-crafted style-shaping reward
weights (``pose``, ``foot_clearance``, ``foot_swing_height``, ``foot_slip``)
are zeroed at construction time so the discriminator carries the full style
signal — see roadmap §2 phase 2 ("AMP replaces style-shaping subset").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from otus_rl_project.algorithms.amp.discriminator import AmpDiscriminator
from otus_rl_project.algorithms.amp.expert_buffer import (
  AMP_FEATURE_DIM,
  AmpExpertBuffer,
  LAFAN1_DEFAULT_FPS,
)
from otus_rl_project.eval.compare_runner import LocomotionCompareVelocityRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

AMP_LOG_PREFIX = "AMP"

STYLE_REWARD_TERMS = ("pose", "foot_clearance", "foot_swing_height", "foot_slip")
"""Reward terms that AMP replaces when ``amp_replace_style_terms=True``."""


def extract_amp_features(menv: ManagerBasedRlEnv) -> torch.Tensor:
  """Match :func:`~otus_rl_project.algorithms.amp.expert_buffer.csv_to_features` channel order.

  Output shape ``(num_envs, 38)`` on ``menv.device``.
  """
  d = menv.scene["robot"].data
  return torch.cat(
    [d.projected_gravity_b, d.root_link_lin_vel_b, d.root_link_ang_vel_b, d.joint_pos],
    dim=-1,
  )


def zero_style_reward_terms(menv: ManagerBasedRlEnv) -> list[str]:
  """In-place set weight=0 on style-shaping reward terms; returns zeroed names.

  ``RewardManager.get_term_cfg`` returns the live :class:`RewardTermCfg` reference,
  so mutating ``.weight`` updates what the manager applies on subsequent steps.
  """
  rm = menv.reward_manager
  zeroed: list[str] = []
  for name in STYLE_REWARD_TERMS:
    try:
      term_cfg = rm.get_term_cfg(name)
    except (KeyError, ValueError):
      continue
    if term_cfg.weight != 0.0:
      term_cfg.weight = 0.0
      zeroed.append(name)
  return zeroed


@dataclass
class AmpCfg:
  """Hyperparameters for :class:`AmpVelocityRunner`."""

  expert_csv_paths: tuple[str, ...] = ()
  """LAFAN1 / openhe-format CSVs (see ``scripts/openhe_pkl_to_csv.py``)."""
  expert_src_fps: float = float(LAFAN1_DEFAULT_FPS)
  expert_target_fps: float | None = None
  """``None`` keeps source rate; runner overrides to the env's policy rate."""

  hidden_dims: tuple[int, ...] = (1024, 512)
  learning_rate: float = 1e-4
  weight_decay: float = 1e-4
  gradient_penalty: float = 5.0
  grad_clip: float = 1.0

  amp_reward_weight: float = 0.5
  """``r = (1-w)·r_task + w·amp_reward_scale·r_style``."""
  amp_reward_scale: float = 2.0
  """Multiplies ``r_style ∈ [0, 1]`` to roughly match per-step task reward magnitude."""
  amp_replace_style_terms: bool = False
  """Zero weights of :data:`STYLE_REWARD_TERMS` so AMP carries the full style signal."""

  disc_batch_size: int = 4096
  disc_updates_per_iter: int = 1
  pair_buffer_capacity: int = 200_000
  """Circular policy-pair buffer (capacity in pairs, not env-steps)."""

  feature_dim: int = AMP_FEATURE_DIM
  """Auto-overridden from the expert buffer at runtime."""

  log_interval: int = 1


@dataclass
class _PolicyPairBuffer:
  """Fixed-capacity ring buffer of (s, s') pairs on ``device``."""

  capacity: int
  feature_dim: int
  device: torch.device | str
  s: torch.Tensor = field(init=False)
  sp: torch.Tensor = field(init=False)
  size: int = 0
  ptr: int = 0

  def __post_init__(self) -> None:
    self.s = torch.zeros((self.capacity, self.feature_dim), device=self.device)
    self.sp = torch.zeros((self.capacity, self.feature_dim), device=self.device)

  def push(self, s: torch.Tensor, sp: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    """Append rows (skip rows where ``mask=False``); wraps when capacity is hit."""
    if mask is not None:
      keep = mask.bool().nonzero(as_tuple=False).flatten()
      if keep.numel() == 0:
        return
      s = s.index_select(0, keep)
      sp = sp.index_select(0, keep)
    n = int(s.shape[0])
    if n == 0:
      return
    cap = self.capacity
    end = self.ptr + n
    if end <= cap:
      self.s[self.ptr : end].copy_(s)
      self.sp[self.ptr : end].copy_(sp)
    else:
      first = cap - self.ptr
      self.s[self.ptr :].copy_(s[:first])
      self.sp[self.ptr :].copy_(sp[:first])
      rem = n - first
      self.s[:rem].copy_(s[first:])
      self.sp[:rem].copy_(sp[first:])
    self.size = min(self.size + n, cap)
    self.ptr = end % cap

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(int(batch_size), self.size)
    idx = torch.randint(0, self.size, (n,), device=self.s.device)
    return self.s[idx], self.sp[idx]


class AmpVelocityRunner(LocomotionCompareVelocityRunner):
  """:class:`LocomotionCompareVelocityRunner` + AMP discriminator + style reward."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    amp_dict = dict(train_cfg.get("amp", {}))
    self.amp_cfg = AmpCfg(**amp_dict)
    self._zeroed_style_terms: list[str] = []
    self._init_amp(device)

    self._last_r_amp_mean: float = 0.0
    self._last_r_amp_max: float = 0.0
    self._last_r_task_mean: float = 0.0

  # ------------------------------------------------------------------ setup

  def _init_amp(self, device: str) -> None:
    if not self.amp_cfg.expert_csv_paths:
      raise ValueError(
        "AmpCfg.expert_csv_paths is empty — pass at least one LAFAN1 CSV to "
        "--amp.expert-csv-paths (relative to repo root or absolute)."
      )
    if self.amp_cfg.amp_replace_style_terms:
      menv = self.env.unwrapped
      self._zeroed_style_terms = zero_style_reward_terms(menv)
      if self._zeroed_style_terms:
        print(
          f"[AMP] zeroed style reward terms: {self._zeroed_style_terms}",
          flush=True,
        )

    target_fps = self.amp_cfg.expert_target_fps
    if target_fps is None:
      target_fps = self._policy_step_fps()

    self.expert_buf = AmpExpertBuffer(
      csv_paths=list(self.amp_cfg.expert_csv_paths),
      device=device,
      src_fps=self.amp_cfg.expert_src_fps,
      target_fps=target_fps,
    )
    print(
      f"[AMP] expert buffer: {len(self.expert_buf)} pairs from "
      f"{len(self.expert_buf.csv_paths)} CSV(s) @ {target_fps:.1f} fps "
      f"(F={self.expert_buf.feature_dim})",
      flush=True,
    )
    self.amp_cfg.feature_dim = self.expert_buf.feature_dim

    self.disc = AmpDiscriminator(
      feature_dim=self.amp_cfg.feature_dim,
      hidden_dims=self.amp_cfg.hidden_dims,
    ).to(device)
    self.disc_opt = torch.optim.Adam(
      self.disc.parameters(),
      lr=self.amp_cfg.learning_rate,
      weight_decay=self.amp_cfg.weight_decay,
    )
    self.policy_buf = _PolicyPairBuffer(
      capacity=int(self.amp_cfg.pair_buffer_capacity),
      feature_dim=int(self.amp_cfg.feature_dim),
      device=device,
    )

  def _policy_step_fps(self) -> float:
    """Effective policy decision rate from env cfg (``1 / (decimation·timestep)``)."""
    cfg = self.env.unwrapped.cfg
    timestep = float(cfg.sim.mujoco.timestep)
    decimation = int(cfg.decimation)
    return 1.0 / (timestep * decimation)

  # ------------------------------------------------------------------ env hook

  def _install_env_step_hook(self) -> None:
    env = self.env
    orig_step = env.step

    def patched_step(actions: torch.Tensor):
      menv = env.unwrapped
      with torch.no_grad():
        s = extract_amp_features(menv)
      obs, rew, dones, extras = orig_step(actions)
      with torch.no_grad():
        sp = extract_amp_features(menv)
        keep_mask = dones == 0
        r_style = self.disc.compute_reward(s, sp)
        r_amp = r_style * self.amp_cfg.amp_reward_scale
        # Done-spanning pairs cross a reset and are nonsense for the discriminator
        # both as training data and as a reward signal.
        r_amp = r_amp * keep_mask.to(r_amp.dtype)
        self._last_r_amp_mean = float(r_amp.mean().item())
        self._last_r_amp_max = float(r_amp.max().item())
        self._last_r_task_mean = float(rew.mean().item())
        self.policy_buf.push(s.detach(), sp.detach(), mask=keep_mask)
      w = self.amp_cfg.amp_reward_weight
      blended = (1.0 - w) * rew + w * r_amp.to(rew.dtype)
      return obs, blended, dones, extras

    env.step = patched_step  # type: ignore[method-assign]
    self._orig_env_step = orig_step

  def _uninstall_env_step_hook(self) -> None:
    if hasattr(self, "_orig_env_step"):
      self.env.step = self._orig_env_step  # type: ignore[method-assign]
      del self._orig_env_step

  # ------------------------------------------------------------------ disc

  def _train_discriminator(self, it: int) -> None:
    if self.policy_buf.size < self.amp_cfg.disc_batch_size:
      return
    log_acc: dict[str, float] = {}
    for _ in range(int(self.amp_cfg.disc_updates_per_iter)):
      s_p, sp_p = self.policy_buf.sample(self.amp_cfg.disc_batch_size)
      s_e, sp_e = self.expert_buf.sample(self.amp_cfg.disc_batch_size)
      losses = self.disc.compute_disc_loss(
        s_p,
        sp_p,
        s_e,
        sp_e,
        lambda_gp=float(self.amp_cfg.gradient_penalty),
      )
      self.disc_opt.zero_grad(set_to_none=True)
      losses["disc_loss"].backward()
      torch.nn.utils.clip_grad_norm_(
        self.disc.parameters(), max_norm=float(self.amp_cfg.grad_clip)
      )
      self.disc_opt.step()
      for k, v in losses.items():
        log_acc[k] = log_acc.get(k, 0.0) + float(v.item())

    n = max(int(self.amp_cfg.disc_updates_per_iter), 1)
    if self.logger.writer is not None and (it % max(self.amp_cfg.log_interval, 1) == 0):
      w = self.logger.writer
      for k, v in log_acc.items():
        w.add_scalar(f"{AMP_LOG_PREFIX}/{k}", v / n, it)
      w.add_scalar(f"{AMP_LOG_PREFIX}/r_amp_mean_last_step", self._last_r_amp_mean, it)
      w.add_scalar(f"{AMP_LOG_PREFIX}/r_amp_max_last_step", self._last_r_amp_max, it)
      w.add_scalar(f"{AMP_LOG_PREFIX}/r_task_mean_last_step", self._last_r_task_mean, it)
      w.add_scalar(f"{AMP_LOG_PREFIX}/policy_pair_buffer_size", float(self.policy_buf.size), it)

  # ------------------------------------------------------------------ learn

  def learn(  # noqa: D102
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self._install_env_step_hook()
    runner = self
    orig_log = self.logger.log

    def amp_wrapped_log(*args, **kwargs) -> None:
      orig_log(*args, **kwargs)  # type: ignore[misc]
      it = int(kwargs.get("it", args[0] if args else 0))
      runner._train_discriminator(it)

    self.logger.log = amp_wrapped_log  # type: ignore[assignment, method-assign]
    try:
      return super().learn(  # type: ignore[no-any-return, misc]
        num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len
      )
    finally:
      self.logger.log = orig_log  # type: ignore[assignment, method-assign]
      self._uninstall_env_step_hook()

  # ------------------------------------------------------------------ checkpoint

  def save(self, path: str, infos=None) -> None:  # noqa: D102
    super().save(path, infos)
    disc_path = self._disc_path(path)
    try:
      torch.save(
        {
          "disc_state_dict": self.disc.state_dict(),
          "disc_opt_state_dict": self.disc_opt.state_dict(),
          "amp_cfg": {
            "feature_dim": self.amp_cfg.feature_dim,
            "hidden_dims": tuple(self.amp_cfg.hidden_dims),
          },
        },
        disc_path,
      )
    except Exception as e:
      print(f"[AMP] discriminator save failed at {disc_path}: {e}", flush=True)

  def load(  # noqa: D102
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    disc_path = self._disc_path(path)
    if os.path.exists(disc_path):
      sd = torch.load(disc_path, map_location=map_location, weights_only=False)
      self.disc.load_state_dict(sd["disc_state_dict"])
      self.disc_opt.load_state_dict(sd["disc_opt_state_dict"])
    return infos

  @staticmethod
  def _disc_path(checkpoint_path: str) -> str:
    if checkpoint_path.endswith(".pt"):
      return checkpoint_path[:-3] + "_disc.pt"
    return checkpoint_path + ".disc"
