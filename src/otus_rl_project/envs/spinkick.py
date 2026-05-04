"""Spinkick task: G1 doing a double spin kick.

Adapted from ``mujocolab/g1_spinkick_example`` (Apache-2.0). Adds an extra
``base_ang_vel_exceed`` termination on top of
:func:`mjlab.tasks.tracking.config.g1.env_cfgs.unitree_g1_flat_tracking_env_cfg`
to keep PPO from discovering trivial ``spin uncontrollably`` policies during
exploration.

The task is registered under the id ``Mjlab-Spinkick-Unitree-G1`` and uses the
standard tracking PPO runner; pass ``--motion-file=.../spinkick.npz`` at
``otus-train`` time to point it at the reference motion.
"""

from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

# Anything beyond this and the policy is producing a "blender" rather than a
# spin kick; episode is terminated so the bad behaviour gets zero return.
_MAX_ANG_VEL = 500 * math.pi / 180.0  # rad/s


def _base_ang_vel_exceed(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


def unitree_g1_spinkick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)
  cfg.terminations["base_ang_vel_exceed"] = TerminationTermCfg(
    func=_base_ang_vel_exceed, params={"threshold": _MAX_ANG_VEL}
  )
  return cfg


def unitree_g1_spinkick_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_spinkick"
  return cfg


register_mjlab_task(
  task_id="Mjlab-Spinkick-Unitree-G1",
  env_cfg=unitree_g1_spinkick_env_cfg(),
  play_env_cfg=unitree_g1_spinkick_env_cfg(play=True),
  rl_cfg=unitree_g1_spinkick_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
