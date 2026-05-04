"""``Otus-G1-Walk-AMP`` task — same env cfg as ``Otus-G1-Walk-Compare``, AMP runner attached.

The training environment is identical to the comparison flat-velocity walk so
the cross-algorithm metrics (``Eval/*``) stay directly comparable. The only
runtime difference comes from the runner: it monkey-patches ``env.step`` to
inject an AMP style reward and trains a discriminator on (s, s') pairs against
LAFAN1 expert clips.

See ``docs/algorithm_comparison_roadmap.md`` §2 phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from otus_rl_project.algorithms.amp.amp_runner import AmpVelocityRunner
from otus_rl_project.algorithms.amp.expert_buffer import LAFAN1_DEFAULT_FPS
from otus_rl_project.envs.locomotion_compare import (
  LocomotionCompareRunnerCfg,
  unitree_g1_walk_compare_env_cfg,
  unitree_g1_walk_compare_rl_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task


@dataclass
class AmpAgentCfg:
  """``--agent.amp.*`` knobs surfaced through tyro.

  ``expert_csv_paths`` accepts repo-relative or absolute paths; the AMP runner
  resamples each clip to the env's policy-step FPS by default.
  """

  expert_csv_paths: tuple[str, ...] = ()
  expert_src_fps: float = float(LAFAN1_DEFAULT_FPS)
  expert_target_fps: float | None = None

  hidden_dims: tuple[int, ...] = (1024, 512)
  learning_rate: float = 1e-4
  weight_decay: float = 1e-4
  gradient_penalty: float = 5.0
  grad_clip: float = 1.0

  amp_reward_weight: float = 0.5
  amp_reward_scale: float = 2.0
  amp_replace_style_terms: bool = False

  disc_batch_size: int = 4096
  disc_updates_per_iter: int = 1
  pair_buffer_capacity: int = 200_000

  log_interval: int = 1


@dataclass
class LocomotionAmpRunnerCfg(LocomotionCompareRunnerCfg):
  """:class:`LocomotionCompareRunnerCfg` + nested ``amp.*`` block (tyro-friendly)."""

  amp: AmpAgentCfg = field(default_factory=AmpAgentCfg)


def unitree_g1_walk_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Same env cfg as ``Otus-G1-Walk-Compare`` (ensures metrics are comparable)."""
  return unitree_g1_walk_compare_env_cfg(play=play)


def unitree_g1_walk_amp_rl_cfg() -> LocomotionAmpRunnerCfg:
  """Compare runner cfg + default ``amp.*`` block; ``experiment_name=g1_walk_amp``."""
  base = unitree_g1_walk_compare_rl_cfg()
  take = {f.name: getattr(base, f.name) for f in fields(LocomotionCompareRunnerCfg)}
  take["experiment_name"] = "g1_walk_amp"
  return LocomotionAmpRunnerCfg(**take)


register_mjlab_task(
  task_id="Otus-G1-Walk-AMP",
  env_cfg=unitree_g1_walk_amp_env_cfg(),
  play_env_cfg=unitree_g1_walk_amp_env_cfg(play=True),
  rl_cfg=unitree_g1_walk_amp_rl_cfg(),
  runner_cls=AmpVelocityRunner,
)
