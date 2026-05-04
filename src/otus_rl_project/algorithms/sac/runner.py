"""Soft Actor-Critic (SAC) training driver mirroring :class:`ArsRunner`'s public surface.

The runner owns the train env, replay buffer, networks, optimizers, and the
shared eval callback (the one PPO/AMP/ARS already use). One iteration =
``rollout_steps`` env steps + ``num_updates_per_iter`` gradient updates +
optional :data:`Eval/*` pass. Termination handling treats all ``dones`` as
"end of trajectory" (truncations and terminations are not separated — minor
bias, kept for parity with the other comparison runners).

References:

- Haarnoja et al., 2018 — *Soft Actor-Critic Algorithms and Applications*
  (with automatic temperature tuning, eq. 17–19).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from otus_rl_project.algorithms.sac.policy import (
  ACTOR_OBS_GROUP,
  GaussianActor,
  ReplayBuffer,
  SacActorAdapter,
  TwinCritic,
)
from otus_rl_project.eval.locomotion_eval import (
  EVAL_PREFIX,
  LocomotionEvalPerturbation,
  log_eval_metrics,
  run_locomotion_eval_rollout,
)
from otus_rl_project.eval.perturbations import clear_pelvis_push

from mjlab.rl import RslRlVecEnvWrapper


@dataclass
class SacCfg:
  """Hyperparameters + eval cadence for :class:`SacRunner`.

  Defaults are tuned for 4096-env mjlab locomotion: small ``rollout_steps``
  + bounded ``num_updates_per_iter`` (UTD ≪ 1) keeps the GPU saturated by
  rollout collection, not by Q-network updates.
  """

  hidden_dims: tuple[int, ...] = (256, 256, 256)
  action_scale: float = 1.0
  """Tanh-squashed action range: env sees ``a ∈ [-action_scale, action_scale]``."""

  gamma: float = 0.99
  tau: float = 0.005
  """Polyak averaging coefficient for soft target updates."""

  actor_lr: float = 3e-4
  critic_lr: float = 3e-4
  alpha_lr: float = 3e-4

  init_log_alpha: float = 0.0
  target_entropy: float | None = None
  """``None`` -> ``-action_dim`` (Haarnoja et al. 2018 default)."""
  autotune_alpha: bool = True
  """When False, ``alpha = exp(init_log_alpha)`` is held constant."""

  rollout_steps: int = 8
  """Env steps per iteration (across all ``num_envs``); buffer fills 4096·8 = 32768/iter at default."""
  num_random_steps: int = 4
  """Iterations of fully random actions before policy rollouts (warmup)."""
  num_updates_per_iter: int = 64
  """Gradient updates per iteration. UTD ≈ ``num_updates_per_iter / (num_envs * rollout_steps)``."""
  batch_size: int = 4096
  replay_buffer_size: int = 500_000

  reward_scale: float = 1.0
  grad_clip: float = 1.0

  eval_interval: int = 25
  eval_num_envs: int = 256
  eval_num_steps: int = 96
  eval_obs_noise_std: float = 0.05
  eval_threshold_mps: float = 0.15

  save_interval: int = 50
  log_interval: int = 1
  seed: int = 42
  experiment_name: str = "g1_walk_compare_sac"
  run_name: str = ""


@dataclass
class _IterStats:
  iteration: int = 0
  total_env_steps: int = 0
  history: list[float] = field(default_factory=list)


def _select_actor_obs(obs: TensorDict | dict | torch.Tensor) -> torch.Tensor:
  if isinstance(obs, (TensorDict, dict)):
    x = obs[ACTOR_OBS_GROUP]
    assert isinstance(x, torch.Tensor)
    return x
  return obs


class SacRunner:
  """SAC driver. Public surface symmetric with :class:`ArsRunner`.

  - :meth:`learn` — train for ``num_iterations`` and return last reward mean.
  - :meth:`save` / :meth:`load` — actor + twin critic + targets + optimizers + α.
  - :meth:`export_policy_to_onnx` — emits the deterministic actor (`tanh(mu)·scale`).
  """

  def __init__(
    self,
    train_env: RslRlVecEnvWrapper,
    eval_env_factory: Callable[[int], RslRlVecEnvWrapper],
    cfg: SacCfg,
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

    self.num_envs = int(train_env.num_envs)
    self.action_dim = int(train_env.num_actions)
    obs_td, _ = train_env.reset()
    actor_obs = _select_actor_obs(obs_td)
    self.obs_dim = int(actor_obs.shape[-1])
    self._cur_obs = obs_td

    torch.manual_seed(cfg.seed)
    self.actor = GaussianActor(
      self.obs_dim,
      self.action_dim,
      hidden_dims=cfg.hidden_dims,
      action_scale=cfg.action_scale,
    ).to(device)
    self.critic = TwinCritic(
      self.obs_dim, self.action_dim, hidden_dims=cfg.hidden_dims
    ).to(device)
    self.critic_target = TwinCritic(
      self.obs_dim, self.action_dim, hidden_dims=cfg.hidden_dims
    ).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    for p in self.critic_target.parameters():
      p.requires_grad_(False)

    self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    self.target_entropy = float(
      cfg.target_entropy if cfg.target_entropy is not None else -self.action_dim
    )
    self.log_alpha = torch.tensor(
      [cfg.init_log_alpha], device=device, dtype=torch.float32, requires_grad=cfg.autotune_alpha
    )
    self.alpha_opt = (
      torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    )

    self.replay = ReplayBuffer(
      capacity=int(cfg.replay_buffer_size),
      actor_obs_dim=self.obs_dim,
      critic_obs_dim=self.obs_dim,
      action_dim=self.action_dim,
      device=device,
    )

    self.writer: SummaryWriter | None = SummaryWriter(log_dir=log_dir)
    self.stats = _IterStats()
    self._first_threshold_logged = False
    self._wall_t0: float | None = None
    self._last_rollout_reward_mean: float = 0.0

  @property
  def alpha(self) -> torch.Tensor:
    return self.log_alpha.exp()

  # ------------------------------------------------------------------ rollout

  def _rollout_collect(self, *, random_actions: bool) -> float:
    """Take ``rollout_steps`` env steps; push transitions; return mean per-step reward.

    During warmup we sample uniform random actions in ``[-action_scale, action_scale]``
    so the buffer has a diverse prefix before the policy is queried.
    """
    obs_td = self._cur_obs
    s_actor = _select_actor_obs(obs_td).to(self.device)
    rew_acc = 0.0
    for _ in range(int(self.cfg.rollout_steps)):
      with torch.no_grad():
        if random_actions:
          a = (
            torch.rand(
              (self.num_envs, self.action_dim), device=self.device, dtype=torch.float32
            )
            * 2.0
            - 1.0
          ) * float(self.cfg.action_scale)
        else:
          a, _ = self.actor(s_actor, deterministic=False)

      next_td, rew, dones, _ = self.train_env.step(a)
      r = rew.float() * float(self.cfg.reward_scale)
      d = dones.float()
      sp_actor = _select_actor_obs(next_td).to(self.device)
      self.replay.push(
        s_actor=s_actor.detach(),
        s_critic=s_actor.detach(),
        a=a.detach(),
        r=r.detach(),
        sp_actor=sp_actor.detach(),
        sp_critic=sp_actor.detach(),
        d=d.detach(),
      )
      rew_acc += float(rew.mean().item())
      s_actor = sp_actor
      obs_td = next_td

    self._cur_obs = obs_td
    self.stats.total_env_steps += int(self.num_envs * self.cfg.rollout_steps)
    return rew_acc / float(self.cfg.rollout_steps)

  # ------------------------------------------------------------------ updates

  def _update_step(self) -> dict[str, float]:
    """One SAC gradient step (critic + actor + α + soft target). Returns scalar logs."""
    s_a, s_c, a, r, sp_a, sp_c, d = self.replay.sample(self.cfg.batch_size)

    with torch.no_grad():
      a_next, log_p_next = self.actor(sp_a, deterministic=False)
      q1_t, q2_t = self.critic_target(sp_c, a_next)
      q_t = torch.min(q1_t, q2_t) - self.alpha * log_p_next
      target = r + self.cfg.gamma * (1.0 - d) * q_t

    q1, q2 = self.critic(s_c, a)
    critic_loss = (q1 - target).pow(2).mean() + (q2 - target).pow(2).mean()
    self.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=float(self.cfg.grad_clip))
    self.critic_opt.step()

    a_pi, log_p_pi = self.actor(s_a, deterministic=False)
    q1_pi, q2_pi = self.critic(s_c, a_pi)
    q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (self.alpha.detach() * log_p_pi - q_pi).mean()
    self.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=float(self.cfg.grad_clip))
    self.actor_opt.step()

    if self.alpha_opt is not None:
      alpha_loss = -(self.log_alpha * (log_p_pi.detach() + self.target_entropy)).mean()
      self.alpha_opt.zero_grad(set_to_none=True)
      alpha_loss.backward()
      self.alpha_opt.step()
      alpha_loss_val = float(alpha_loss.item())
    else:
      alpha_loss_val = 0.0

    with torch.no_grad():
      tau = float(self.cfg.tau)
      for p, pt in zip(self.critic.parameters(), self.critic_target.parameters(), strict=False):
        pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    return {
      "critic_loss": float(critic_loss.item()),
      "actor_loss": float(actor_loss.item()),
      "alpha_loss": alpha_loss_val,
      "alpha": float(self.alpha.item()),
      "entropy": float(-log_p_pi.detach().mean().item()),
      "q1_mean": float(q1.detach().mean().item()),
      "q2_mean": float(q2.detach().mean().item()),
      "target_q_mean": float(target.detach().mean().item()),
    }

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
    actor = SacActorAdapter(self.actor)

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
        "actor_state_dict": self.actor.state_dict(),
        "critic_state_dict": self.critic.state_dict(),
        "critic_target_state_dict": self.critic_target.state_dict(),
        "actor_opt_state_dict": self.actor_opt.state_dict(),
        "critic_opt_state_dict": self.critic_opt.state_dict(),
        "log_alpha": self.log_alpha.detach().cpu(),
        "alpha_opt_state_dict": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
        "iteration": self.stats.iteration,
        "total_env_steps": self.stats.total_env_steps,
        "obs_dim": self.obs_dim,
        "action_dim": self.action_dim,
      },
      path,
    )

  def load(self, path: str) -> None:  # noqa: D102
    sd = torch.load(path, map_location=self.device, weights_only=False)
    self.actor.load_state_dict(sd["actor_state_dict"])
    self.critic.load_state_dict(sd["critic_state_dict"])
    self.critic_target.load_state_dict(sd["critic_target_state_dict"])
    self.actor_opt.load_state_dict(sd["actor_opt_state_dict"])
    self.critic_opt.load_state_dict(sd["critic_opt_state_dict"])
    self.log_alpha.data.copy_(sd["log_alpha"].to(self.device))
    if self.alpha_opt is not None and sd.get("alpha_opt_state_dict") is not None:
      self.alpha_opt.load_state_dict(sd["alpha_opt_state_dict"])
    self.stats.iteration = int(sd.get("iteration", 0))
    self.stats.total_env_steps = int(sd.get("total_env_steps", 0))

  def export_policy_to_onnx(self, dir_path: str, filename: str = "policy.onnx") -> str:
    """Emit the deterministic actor (``tanh(μ)·scale``) as ``policy.onnx``."""
    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, filename)

    cpu_actor = GaussianActor(
      self.obs_dim,
      self.action_dim,
      hidden_dims=tuple(self.cfg.hidden_dims),
      action_scale=self.cfg.action_scale,
    ).to("cpu").eval()
    cpu_actor.load_state_dict({k: v.detach().cpu() for k, v in self.actor.state_dict().items()})

    class _Det(torch.nn.Module):
      def __init__(self, inner: GaussianActor) -> None:
        super().__init__()
        self.inner = inner

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_deterministic(x)

    dummy = torch.zeros(1, self.obs_dim)
    torch.onnx.export(
      _Det(cpu_actor),
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
    n_warmup = int(self.cfg.num_random_steps)

    for it in range(1, num_iterations + 1):
      self.stats.iteration = it
      use_random = it <= n_warmup
      mean_rew = self._rollout_collect(random_actions=use_random)
      self._last_rollout_reward_mean = mean_rew

      n_updates = 0
      logs_acc: dict[str, float] = {}
      if (not use_random) and self.replay.size >= self.cfg.batch_size:
        for _ in range(int(self.cfg.num_updates_per_iter)):
          step_logs = self._update_step()
          for k, v in step_logs.items():
            logs_acc[k] = logs_acc.get(k, 0.0) + v
          n_updates += 1

      self.stats.history.append(mean_rew)
      last_mean = mean_rew

      if self.writer is not None and (it % self.cfg.log_interval == 0):
        w = self.writer
        w.add_scalar("Train/mean_step_reward", mean_rew, it)
        w.add_scalar("Train/total_env_steps", self.stats.total_env_steps, it)
        w.add_scalar(
          "Train/wall_clock_seconds",
          time.monotonic() - (self._wall_t0 or time.monotonic()),
          it,
        )
        w.add_scalar("Train/replay_buffer_size", float(self.replay.size), it)
        if n_updates > 0:
          for k, v in logs_acc.items():
            w.add_scalar(f"SAC/{k}", v / n_updates, it)

      self._run_eval_phase(it)

      if self.cfg.save_interval > 0 and (it % self.cfg.save_interval == 0):
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      tag = "warmup" if use_random else "train"
      print(
        f"[SAC:{tag}] it={it:5d}  mean_step_R={mean_rew:+.4f}  "
        f"updates={n_updates}  buf={self.replay.size}  "
        f"alpha={float(self.alpha.item()):.4f}  steps={self.stats.total_env_steps}",
        flush=True,
      )

    final_pt = os.path.join(self.log_dir, f"model_{self.stats.iteration}.pt")
    self.save(final_pt)
    try:
      self.export_policy_to_onnx(self.log_dir, "policy.onnx")
    except Exception as e:
      print(f"[SAC] ONNX export failed (training continues): {e}", flush=True)

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


def build_eval_env_factory(train_env: RslRlVecEnvWrapper) -> Callable[[int], RslRlVecEnvWrapper]:
  """Closure that builds a smaller eval vec env with the same cfg (frozen, fresh sim).

  Identical helper to the one in ``algorithms.ars.runner`` — kept local so SAC
  doesn't import from ARS just to share four lines.
  """
  from dataclasses import replace

  from mjlab.envs import ManagerBasedRlEnv

  def factory(num_envs: int) -> RslRlVecEnvWrapper:
    src = train_env.unwrapped
    n = min(int(num_envs), int(src.num_envs))
    new_cfg = replace(src.cfg, scene=replace(src.cfg.scene, num_envs=n))
    dev = str(src.device) if not isinstance(src.device, str) else src.device
    menv = ManagerBasedRlEnv(cfg=new_cfg, device=dev, render_mode=None)
    return RslRlVecEnvWrapper(menv, clip_actions=getattr(train_env, "clip_actions", None))

  return factory
