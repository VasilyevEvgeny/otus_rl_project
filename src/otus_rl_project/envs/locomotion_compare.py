"""Frozen G1 flat velocity task for cross-algorithm comparison (see ``docs/algorithm_comparison_roadmap.md``).

The environment matches ``Mjlab-Velocity-Flat-Unitree-G1``; the runner adds periodic
``Eval/*`` logging on a **separate** small vector env.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from otus_rl_project.eval.compare_runner import LocomotionCompareVelocityRunner


@dataclass
class LocomotionCompareRunnerCfg(RslRlOnPolicyRunnerCfg):
  """G1 PPO + ``LocomotionCompareVelocityRunner`` (tyro: ``--agent.locomotion-eval-interval=...``)."""

  locomotion_eval_interval: int = 50
  """0 = disable. One tick runs nominal, obs-noise, 5N, and 15N passes (sequential)."""
  locomotion_eval_num_steps: int = 96
  locomotion_eval_num_envs: int = 256
  locomotion_eval_threshold_mps: float = 0.15
  """Sample-efficiency threshold for ``Eval/env_steps_to_threshold``."""
  locomotion_eval_obs_noise_std: float = 0.05


def unitree_g1_walk_compare_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """:class:`ManagerBasedRlEnvCfg` matches ``Mjlab-Velocity-Flat-Unitree-G1``."""
  return unitree_g1_flat_env_cfg(play=play)


def unitree_g1_walk_compare_rl_cfg() -> LocomotionCompareRunnerCfg:
  """Same hyperparameters as ``unitree_g1_ppo_runner_cfg`` + compare defaults + ``g1_walk_compare`` dir."""
  b = unitree_g1_ppo_runner_cfg()
  take = {f.name: getattr(b, f.name) for f in fields(RslRlOnPolicyRunnerCfg)}
  take["experiment_name"] = "g1_walk_compare"
  return LocomotionCompareRunnerCfg(**take)


register_mjlab_task(
  task_id="Otus-G1-Walk-Compare",
  env_cfg=unitree_g1_walk_compare_env_cfg(),
  play_env_cfg=unitree_g1_walk_compare_env_cfg(play=True),
  rl_cfg=unitree_g1_walk_compare_rl_cfg(),
  runner_cls=LocomotionCompareVelocityRunner,
)
