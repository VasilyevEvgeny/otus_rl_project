"""DoubleKong task: G1 doing a high-dynamic run + dive flip + recovery.

Adapted from the same MimicKit family as the spinkick example. The motion
``g1_double_kong.pkl`` (7.2 s, 12.4 m of forward translation, peak pelvis
height 1.15 m, lead-leg foot reaching 1.44 m during inversion) is *much*
more aggressive than the canonical LAFAN1 / spinkick clips, and PPO with the
default tracking termination thresholds plateaus around ``mean_reward ~ 3``
because it cannot survive the inversion phase under tight ``z``-only
position/orientation tolerances.

To unstick training we apply the same trick that the OmniRetarget paper
used for the parkour-tic-tac-backflip wall-flip on G1: relax the
end-effector position threshold and effectively disable the anchor
orientation termination (which fires every time the torso projected gravity
crosses the horizon during the flip).

Task id: ``Mjlab-DoubleKong-Unitree-G1``. Pass
``--motion-file=.../double_kong.npz`` at ``otus-train`` time.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

# Default tracking thresholds (from mjlab.tasks.tracking.tracking_env_cfg):
#   ee_body_pos:  0.25  m  (z-only on ankles + wrists)
#   anchor_pos:   0.25  m  (z-only on torso link)
#   anchor_ori:   0.8        (gravity projection diff, max ~2.0)
#
# For double_kong we relax to:
_EE_BODY_POS_THRESHOLD = 0.6   # foot inverts up to ~1.4 m above pelvis
_ANCHOR_POS_THRESHOLD = 0.6    # pelvis swings 0.65 → 1.15 m
_ANCHOR_ORI_THRESHOLD = 1.6    # near-max — effectively disabled during flip


def unitree_g1_double_kong_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  if "ee_body_pos" in cfg.terminations:
    cfg.terminations["ee_body_pos"].params["threshold"] = _EE_BODY_POS_THRESHOLD
  if "anchor_pos" in cfg.terminations:
    cfg.terminations["anchor_pos"].params["threshold"] = _ANCHOR_POS_THRESHOLD
  if "anchor_ori" in cfg.terminations:
    cfg.terminations["anchor_ori"].params["threshold"] = _ANCHOR_ORI_THRESHOLD

  return cfg


def unitree_g1_double_kong_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_double_kong"
  return cfg


register_mjlab_task(
  task_id="Mjlab-DoubleKong-Unitree-G1",
  env_cfg=unitree_g1_double_kong_env_cfg(),
  play_env_cfg=unitree_g1_double_kong_env_cfg(play=True),
  rl_cfg=unitree_g1_double_kong_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
