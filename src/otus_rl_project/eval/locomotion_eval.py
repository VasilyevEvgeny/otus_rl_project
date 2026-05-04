"""Reward-independent locomotion metrics for cross-algorithm comparison (roadmap §4).

Scalar tags use prefix :const:`EVAL_PREFIX` so TensorBoard / W&B plots align across algorithms.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from otus_rl_project.eval.perturbations import (
  add_tensordict_gaussian_noise_,
  apply_pelvis_horizontal_push_n,
  clear_pelvis_push,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

EVAL_PREFIX = "Eval"

EVAL_NOISE_SEED = 123_456_789


class LocomotionEvalPerturbation(Enum):
  """Robustness setting for :func:`run_locomotion_eval_rollout`."""

  NONE = auto()
  OBS_NOISE = auto()
  PUSH_5N = auto()
  PUSH_15N = auto()


def _feet_slip_metric(menv: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean foot slip speed (m/s) when feet have ground contact (batch mean)."""
  robot = menv.scene["robot"]
  try:
    contact_sensor = menv.scene["feet_ground_contact"]
  except KeyError:
    return torch.tensor(0.0, device=menv.device)
  command = menv.command_manager.get_command("twist")
  if command is None:
    return torch.tensor(0.0, device=menv.device)
  site_names = ("left_foot", "right_foot")
  s_ids, _ = robot.find_sites(site_names, preserve_order=True)
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = robot.data.site_lin_vel_w[:, s_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  num_in = torch.clamp(torch.sum(in_contact), min=1.0)
  return torch.sum(vel_xy_norm * in_contact) / num_in


def _step_scalar_metrics(
  menv: ManagerBasedRlEnv,
  prev_action: torch.Tensor | None,
  action: torch.Tensor,
) -> dict[str, float]:
  robot = menv.scene["robot"]
  cmd = menv.command_manager.get_command("twist")
  assert cmd is not None
  v_xy_cmd = cmd[:, :2]
  v_xy = robot.data.root_link_lin_vel_b[:, :2]
  vel_l2 = torch.mean(torch.norm(v_xy_cmd - v_xy, dim=1))

  w_cmd = cmd[:, 2]
  wz = robot.data.root_link_ang_vel_b[:, 2]
  yaw_l1 = torch.mean(torch.abs(w_cmd - wz))

  h = robot.data.root_link_pos_w[:, 2]
  h_std = torch.std(h)

  ap = robot.data.actuator_force
  tc = torch.mean(torch.sum(torch.square(ap), dim=1))

  sm = (
    torch.mean(torch.sum(torch.square(action - prev_action), dim=1))
    if prev_action is not None
    else torch.tensor(0.0, device=menv.device)
  )
  fs = _feet_slip_metric(menv)
  return {
    f"{EVAL_PREFIX}/velocity_tracking_error_xy": float(vel_l2.item()),
    f"{EVAL_PREFIX}/yaw_tracking_error": float(yaw_l1.item()),
    f"{EVAL_PREFIX}/base_height_std": float(h_std.item()),
    f"{EVAL_PREFIX}/torque_cost": float(tc.item()),
    f"{EVAL_PREFIX}/action_smoothness": float(sm.item()),
    f"{EVAL_PREFIX}/foot_slippage": float(fs.item()),
  }


def _sum_dict_into(acc: dict[str, float], add: dict[str, float]) -> None:
  for k, v in add.items():
    acc[k] = acc.get(k, 0.0) + v


def _mean_dict(accum: dict[str, float], n: int) -> dict[str, float]:
  if n <= 0:
    return {k: 0.0 for k in accum}
  return {k: v / float(n) for k, v in accum.items()}


def run_locomotion_eval_rollout(
  actor: Any,
  eval_vec_env: Any,
  device: str,
  *,
  num_steps: int = 96,
  perturbation: LocomotionEvalPerturbation = LocomotionEvalPerturbation.NONE,
  obs_noise_std: float = 0.05,
) -> dict[str, float]:
  """Roll out a deterministic policy on a **dedicated** eval vec env; returns **time-averaged** metrics.

  Does not call PPO :meth:`~rsl_rl.algorithms.ppo.PPO.process_env_step` (no training-side
  normalizer or buffer updates).
  """
  menv: ManagerBasedRlEnv = eval_vec_env.unwrapped
  g_cpu = torch.Generator(device="cpu")
  g_cpu.manual_seed(EVAL_NOISE_SEED)

  actor.eval()
  with torch.inference_mode():
    obs, _ = eval_vec_env.reset()
    obs = obs.to(device)
    if (
      isinstance(obs, TensorDict)
      and perturbation is LocomotionEvalPerturbation.OBS_NOISE
    ):
      add_tensordict_gaussian_noise_(obs, obs_noise_std, generator=g_cpu)

    prev: torch.Tensor | None = None
    accum: dict[str, float] = {}
    ep_len_sum = 0.0

    for t in range(num_steps):
      if perturbation in (
        LocomotionEvalPerturbation.PUSH_5N,
        LocomotionEvalPerturbation.PUSH_15N,
      ) and (t % 48) < 4:
        n = 5.0 if perturbation is LocomotionEvalPerturbation.PUSH_5N else 15.0
        apply_pelvis_horizontal_push_n(menv, newtons=n, world_axis=0)
      else:
        clear_pelvis_push(menv)

      action = actor(obs, stochastic_output=False)
      next_obs, _, _, _ = eval_vec_env.step(action)
      mets = _step_scalar_metrics(menv, prev, action)
      _sum_dict_into(accum, mets)
      prev = action

      el = menv.episode_length_buf.float().mean()
      ep_len_sum += float(el.item())

      obs = next_obs.to(device)
      if (
        isinstance(obs, TensorDict)
        and perturbation is LocomotionEvalPerturbation.OBS_NOISE
      ):
        add_tensordict_gaussian_noise_(obs, obs_noise_std, generator=g_cpu)

    clear_pelvis_push(menv)
  out = _mean_dict(accum, num_steps)
  if num_steps > 0:
    out[f"{EVAL_PREFIX}/episode_length"] = ep_len_sum / float(num_steps)
  return out


def log_eval_metrics(
  writer: Any, metrics: dict[str, float], step: int, key_suffix: str = ""
) -> None:
  """Write ``Eval/*`` keys. When ``key_suffix`` is set, tags become ``Eval/..._{suffix}`` (no double slash)."""
  for k, v in metrics.items():
    if not k.startswith(f"{EVAL_PREFIX}/"):
      continue
    if key_suffix:
      sub = k[len(f"{EVAL_PREFIX}/") :]
      tag = f"{EVAL_PREFIX}/{sub}_{key_suffix}"
    else:
      tag = k
    writer.add_scalar(tag, v, step)
