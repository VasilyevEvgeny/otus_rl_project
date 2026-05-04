"""Observation noise and push forces for robustness eval (§4.2 of the comparison roadmap).

These helpers operate on a :class:`rsl_rl.env.vec_env.VecEnv` (e.g. :class:`mjlab.rl.RslRlVecEnvWrapper`)
*without* mutating the training env: use them only with the dedicated eval environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def add_tensordict_gaussian_noise_(
  obs: TensorDict,
  std: float,
  generator: torch.Generator | None = None,
) -> None:
  """In-place: add N(0, std²) to every float tensor in a TensorDict (typical PPO obs group).

  ``generator`` may live on a different device than the tensor (e.g. a seeded CPU
  generator with CUDA observations); we sample on the generator's device and copy.
  """
  if std <= 0.0:
    return
  for v in obs.values():
    if isinstance(v, TensorDict):
      add_tensordict_gaussian_noise_(v, std, generator=generator)
    elif isinstance(v, torch.Tensor) and v.is_floating_point():
      gen_dev = generator.device if generator is not None else v.device
      if gen_dev == v.device:
        noise = torch.randn(
          v.shape, device=v.device, dtype=v.dtype, generator=generator
        ) * std
      else:
        noise = (
          torch.randn(v.shape, device=gen_dev, dtype=torch.float32, generator=generator)
          * std
        ).to(device=v.device, dtype=v.dtype)
      v.add_(noise)


def clear_pelvis_push(menv: ManagerBasedRlEnv) -> None:
  """Zero external wrenches on ``torso_link`` (G1) so the next sim step is unperturbed."""
  robot = menv.scene["robot"]
  num_envs = menv.num_envs
  body_ids, _ = robot.find_bodies("torso_link")
  if not body_ids:
    return
  b = len(body_ids)
  dev = menv.device
  zf = torch.zeros((num_envs, b, 3), device=dev, dtype=torch.float32)
  zt = torch.zeros_like(zf)
  robot.write_external_wrench_to_sim(zf, zt, body_ids=body_ids)


def apply_pelvis_horizontal_push_n(
  menv: ManagerBasedRlEnv,
  newtons: float,
  world_axis: int = 0,
) -> None:
  """Constant horizontal push on ``torso_link`` CoM in world frame (N), until cleared.

  ``world_axis`` 0,1,2 = X,Y,Z. Forward push on flat ground: ``world_axis=0`` (X).
  """
  robot = menv.scene["robot"]
  num_envs = menv.num_envs
  body_ids, _ = robot.find_bodies("torso_link")
  if not body_ids:
    return
  b = len(body_ids)
  dev = menv.device
  forces = torch.zeros((num_envs, b, 3), device=dev, dtype=torch.float32)
  forces[:, :, world_axis] = newtons
  torques = torch.zeros_like(forces)
  robot.write_external_wrench_to_sim(forces, torques, body_ids=body_ids)
