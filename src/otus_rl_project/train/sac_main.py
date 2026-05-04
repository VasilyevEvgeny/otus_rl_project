"""``otus-train-sac`` — Soft Actor-Critic on the comparison locomotion task.

Mirrors the :mod:`otus_rl_project.train.ars_main` pattern (own :mod:`tyro` CLI,
no upstream delegation) because SAC is off-policy and reuses only the env
builder + eval callback from the rest of the project.

Usage examples (inside the container)::

    otus-train-sac
    otus-train-sac --num-envs 4096 --num-iterations 1000
    otus-train-sac --num-iterations 2000 --num-updates-per-iter 32 --batch-size 4096
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import torch
import tyro

from otus_rl_project.algorithms.sac.runner import (
  SacCfg,
  SacRunner,
  build_eval_env_factory,
)
from otus_rl_project.utils.upstream import PROJECT_ROOT, bootstrap_env

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg


@dataclass
class SacCliConfig:
  """Frozen tyro CLI for :func:`main`. Only SAC-relevant knobs are exposed."""

  task_id: Literal["Otus-G1-Walk-Compare"] = "Otus-G1-Walk-Compare"
  """Comparison-frozen flat-velocity task (single choice, by design)."""
  num_envs: int = 4096
  num_iterations: int = 1000
  device: str = "cuda:0"
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  hidden_dims: tuple[int, ...] = (256, 256, 256)
  action_scale: float = 1.0

  gamma: float = 0.99
  tau: float = 0.005
  actor_lr: float = 3e-4
  critic_lr: float = 3e-4
  alpha_lr: float = 3e-4
  init_log_alpha: float = 0.0
  target_entropy: float | None = None
  autotune_alpha: bool = True

  rollout_steps: int = 8
  num_random_steps: int = 4
  num_updates_per_iter: int = 64
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


def _select_device(gpu_ids: list[int] | Literal["all"] | None, fallback: str) -> str:
  if gpu_ids is None or gpu_ids == "all":
    return fallback
  if not isinstance(gpu_ids, list) or not gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu"
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:  # noqa: D103
  bootstrap_env()
  cfg = tyro.cli(SacCliConfig)

  device = _select_device(cfg.gpu_ids, cfg.device)
  os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

  env_cfg = load_env_cfg(cfg.task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed

  log_root = PROJECT_ROOT / "logs" / "rsl_rl" / cfg.experiment_name
  ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = log_root / (f"{ts}_{cfg.run_name}" if cfg.run_name else ts)
  log_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] task={cfg.task_id} device={device} num_envs={cfg.num_envs}")
  print(f"[INFO] log_dir={log_dir}")

  menv = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  train_env = RslRlVecEnvWrapper(menv, clip_actions=cfg.action_scale)

  runner_cfg = SacCfg(
    hidden_dims=tuple(cfg.hidden_dims),
    action_scale=cfg.action_scale,
    gamma=cfg.gamma,
    tau=cfg.tau,
    actor_lr=cfg.actor_lr,
    critic_lr=cfg.critic_lr,
    alpha_lr=cfg.alpha_lr,
    init_log_alpha=cfg.init_log_alpha,
    target_entropy=cfg.target_entropy,
    autotune_alpha=cfg.autotune_alpha,
    rollout_steps=cfg.rollout_steps,
    num_random_steps=cfg.num_random_steps,
    num_updates_per_iter=cfg.num_updates_per_iter,
    batch_size=cfg.batch_size,
    replay_buffer_size=cfg.replay_buffer_size,
    reward_scale=cfg.reward_scale,
    grad_clip=cfg.grad_clip,
    eval_interval=cfg.eval_interval,
    eval_num_envs=cfg.eval_num_envs,
    eval_num_steps=cfg.eval_num_steps,
    eval_obs_noise_std=cfg.eval_obs_noise_std,
    eval_threshold_mps=cfg.eval_threshold_mps,
    save_interval=cfg.save_interval,
    log_interval=cfg.log_interval,
    seed=cfg.seed,
    experiment_name=cfg.experiment_name,
    run_name=cfg.run_name,
  )

  runner = SacRunner(
    train_env=train_env,
    eval_env_factory=build_eval_env_factory(train_env),
    cfg=runner_cfg,
    log_dir=str(log_dir),
    device=device,
  )

  try:
    final = runner.learn(num_iterations=cfg.num_iterations)
    print(f"[SAC] DONE. final mean_step_R={final:+.4f}")
  finally:
    train_env.close()


if __name__ == "__main__":
  main()
