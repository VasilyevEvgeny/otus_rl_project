"""``otus-train-ars`` — Augmented Random Search (V2-t) on the comparison locomotion task.

Owns its own :mod:`tyro` CLI (no upstream delegation) because ARS is gradient-free
and reuses only the env builder + eval callback from the rest of the project.

Usage examples (inside the container)::

    otus-train-ars
    otus-train-ars --task-id Otus-G1-Walk-Compare --num-iterations 500
    otus-train-ars --num-envs 4096 --num-directions 64 --top-directions 32
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro

from otus_rl_project.algorithms.ars.runner import (
  ArsRunner,
  ArsRunnerCfg,
  build_eval_env_factory,
)
from otus_rl_project.utils.upstream import PROJECT_ROOT, bootstrap_env

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg


@dataclass
class ArsCliConfig:
  """Frozen tyro CLI for :func:`main`. Only ARS-relevant knobs are exposed."""

  task_id: Literal["Otus-G1-Walk-Compare"] = "Otus-G1-Walk-Compare"
  """Comparison-frozen flat-velocity task (single choice, by design)."""
  num_envs: int = 4096
  """Must equal ``num_directions × 2 × envs_per_direction``."""
  num_iterations: int = 500
  device: str = "cuda:0"
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  num_directions: int = 64
  top_directions: int = 32
  rollout_steps: int = 200
  step_size: float = 0.02
  noise_std: float = 0.03
  clip_actions: float = 4.0
  reward_to_go_clip: float | None = None

  eval_interval: int = 25
  eval_num_envs: int = 256
  eval_num_steps: int = 96
  eval_obs_noise_std: float = 0.05
  eval_threshold_mps: float = 0.15

  save_interval: int = 50
  log_interval: int = 1
  seed: int = 42
  experiment_name: str = "g1_walk_compare_ars"
  run_name: str = ""


def _select_device(gpu_ids: list[int] | Literal["all"] | None, fallback: str) -> str:
  """Set ``CUDA_VISIBLE_DEVICES`` from ``gpu_ids`` and return the torch device string."""
  if gpu_ids is None or gpu_ids == "all":
    return fallback
  if not isinstance(gpu_ids, list) or not gpu_ids:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu"
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:  # noqa: D103
  bootstrap_env()
  cfg = tyro.cli(ArsCliConfig)

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
  train_env = RslRlVecEnvWrapper(menv, clip_actions=cfg.clip_actions)

  runner_cfg = ArsRunnerCfg(
    num_directions=cfg.num_directions,
    top_directions=cfg.top_directions,
    rollout_steps=cfg.rollout_steps,
    step_size=cfg.step_size,
    noise_std=cfg.noise_std,
    clip_actions=cfg.clip_actions,
    reward_to_go_clip=cfg.reward_to_go_clip,
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

  runner = ArsRunner(
    train_env=train_env,
    eval_env_factory=build_eval_env_factory(train_env),
    cfg=runner_cfg,
    log_dir=str(log_dir),
    device=device,
  )

  try:
    final = runner.learn(num_iterations=cfg.num_iterations)
    print(f"[ARS] DONE. final mean_R={final:+.3f}")
  finally:
    train_env.close()


if __name__ == "__main__":
  main()
