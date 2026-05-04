"""PPO runner with periodic locomotion eval on a **separate** vector env (training sim untouched)."""

from __future__ import annotations

import time
from dataclasses import replace

from rsl_rl.env.vec_env import VecEnv

from otus_rl_project.eval.locomotion_eval import (
  EVAL_PREFIX,
  LocomotionEvalPerturbation,
  log_eval_metrics,
  run_locomotion_eval_rollout,
)
from otus_rl_project.eval.perturbations import clear_pelvis_push

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def _build_eval_vec_env(
  train_env: RslRlVecEnvWrapper,
  *,
  num_envs: int,
) -> RslRlVecEnvWrapper:
  """Smaller (or same) vector env, identical config — training state stays untouched."""
  src = train_env.unwrapped
  c = src.cfg
  new_cfg = replace(c, scene=replace(c.scene, num_envs=int(num_envs)))
  dev = str(src.device) if not isinstance(src.device, str) else src.device
  menv = ManagerBasedRlEnv(cfg=new_cfg, device=dev, render_mode=None)
  return RslRlVecEnvWrapper(
    menv, clip_actions=getattr(train_env, "clip_actions", None)
  )


def _vel_xy_key() -> str:
  return f"{EVAL_PREFIX}/velocity_tracking_error_xy"


class LocomotionCompareVelocityRunner(VelocityOnPolicyRunner):
  """:class:`VelocityOnPolicyRunner` + roadmap §4 eval (``Eval/*`` in TB / W&B)."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._eval_vec: RslRlVecEnvWrapper | None = None
    self._first_threshold_logged = False
    self._wall_t0: float | None = None

  def _ensure_eval_vec(self) -> RslRlVecEnvWrapper:
    if self._eval_vec is None:
      req = int(self.cfg.get("locomotion_eval_num_envs", 256))
      cap = int(self.env.num_envs)
      n = min(req, cap)
      self._eval_vec = _build_eval_vec_env(self.env, num_envs=n)
    return self._eval_vec

  def _run_one_eval_phase(self, it: int) -> None:
    assert self.logger.writer is not None
    w = self.logger.writer
    n_steps = int(self.cfg.get("locomotion_eval_num_steps", 96))
    n_std = float(self.cfg.get("locomotion_eval_obs_noise_std", 0.05))
    ev = self._ensure_eval_vec()
    actor = self.alg.get_policy()
    dev = str(self.device) if not isinstance(self.device, str) else self.device

    seq = [
      (LocomotionEvalPerturbation.NONE, "", {}),
      (LocomotionEvalPerturbation.OBS_NOISE, "obs_noise", {"obs_noise_std": n_std}),
      (LocomotionEvalPerturbation.PUSH_5N, "push_5N", {}),
      (LocomotionEvalPerturbation.PUSH_15N, "push_15N", {}),
    ]
    for pert, suffix, extra in seq:
      m = run_locomotion_eval_rollout(
        actor,
        ev,
        dev,
        num_steps=n_steps,
        perturbation=pert,
        **extra,
      )
      log_eval_metrics(w, m, it, key_suffix=suffix)
      v_err = m.get(_vel_xy_key(), 0.0)
      if pert is LocomotionEvalPerturbation.OBS_NOISE:
        w.add_scalar(f"{EVAL_PREFIX}/vel_error_obs_noise", v_err, it)
      elif pert is LocomotionEvalPerturbation.PUSH_5N:
        w.add_scalar(f"{EVAL_PREFIX}/vel_error_push_5N", v_err, it)
      elif pert is LocomotionEvalPerturbation.PUSH_15N:
        w.add_scalar(f"{EVAL_PREFIX}/vel_error_push_15N", v_err, it)
      if not self._first_threshold_logged and pert is LocomotionEvalPerturbation.NONE:
        thr = float(self.cfg.get("locomotion_eval_threshold_mps", 0.15))
        if m.get(_vel_xy_key(), 1e6) < thr:
          self._first_threshold_logged = True
          w.add_scalar(  # type: ignore[union-attr]
            f"{EVAL_PREFIX}/env_steps_to_threshold",
            int(self.logger.tot_timesteps),
            it,
          )
          w.add_scalar(
            f"{EVAL_PREFIX}/wall_clock_to_threshold",
            time.monotonic() - float(self._wall_t0 or time.monotonic()),
            it,
          )

  def _maybe_locomotion_eval(self, it: int) -> None:
    if getattr(self.logger, "disable_logs", False) or self.logger.writer is None:
      return
    every = int(self.cfg.get("locomotion_eval_interval", 0))
    if every <= 0 or (it % every) != 0 or it < 0:
      return
    if self._wall_t0 is None:
      self._wall_t0 = time.monotonic()
    self._run_one_eval_phase(it)

  def learn(  # noqa: D102
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if self._wall_t0 is None:
      self._wall_t0 = time.monotonic()
    r = self
    _orig = self.logger.log

    def _wrapped(*args, **kwargs) -> None:
      _orig(*args, **kwargs)  # type: ignore[misc]
      it = int(kwargs.get("it", args[0] if args else 0))
      r._maybe_locomotion_eval(it)

    self.logger.log = _wrapped  # type: ignore[assignment, method-assign]
    try:
      return super().learn(  # type: ignore[no-any-return, misc]
        num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len
      )
    finally:
      self.logger.log = _orig  # type: ignore[assignment, method-assign]
      if self._eval_vec is not None:
        try:
          clear_pelvis_push(self._eval_vec.unwrapped)  # type: ignore[union-attr]
        except Exception:  # noqa: S110
          pass
        self._eval_vec.close()
        self._eval_vec = None
