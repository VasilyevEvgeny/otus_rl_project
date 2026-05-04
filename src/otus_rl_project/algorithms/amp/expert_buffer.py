"""LAFAN1 CSV → AMP expert (s, s') buffer.

CSV layout (one row = one frame, see ``scripts/openhe_pkl_to_csv.py``)::

    [x, y, z,  qx, qy, qz, qw,  dof_0 .. dof_28]   # 7 + 29 = 36 columns

We turn each frame into a 38-D AMP feature vector matching what the discriminator
sees from sim-side rollouts (see :mod:`otus_rl_project.algorithms.amp.amp_runner`)::

    f = [ projected_gravity_b (3),
          root_link_lin_vel_b (3),
          root_link_ang_vel_b (3),
          joint_pos           (29) ]

Linear / angular velocities are computed via finite differences on the CSV
positions / quaternions and re-projected into the body frame so they carry the
same physical units (m/s, rad/s) as ``robot.data.*``. The dataset is optionally
resampled to a target FPS so the (s, s') time gap matches the policy step ``dt``
(decimation × ``cfg.sim.timestep``); the default keeps the LAFAN1 native rate.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

GRAVITY_W = np.array([0.0, 0.0, -9.81], dtype=np.float64)
"""World-frame gravity vector (matches ``mjlab.entity.data.gravity_vec_w``)."""

LAFAN1_DEFAULT_FPS = 30
"""Native sample rate of LAFAN1 / openhe G1-retargeted CSVs."""

NUM_DOF_G1 = 29

AMP_FEATURE_DIM = 3 + 3 + 3 + NUM_DOF_G1  # = 38


def _quat_apply(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate ``v`` by quaternion ``q`` (xyzw convention), Hamilton product."""
  qx, qy, qz, qw = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
  vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
  tx = 2.0 * (qy * vz - qz * vy)
  ty = 2.0 * (qz * vx - qx * vz)
  tz = 2.0 * (qx * vy - qy * vx)
  rx = vx + qw * tx + (qy * tz - qz * ty)
  ry = vy + qw * ty + (qz * tx - qx * tz)
  rz = vz + qw * tz + (qx * ty - qy * tx)
  return np.stack([rx, ry, rz], axis=-1)


def _quat_apply_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate ``v`` by inverse of ``q`` (mirrors ``mjlab.utils.math.quat_apply_inverse``)."""
  q_conj = q_xyzw.copy()
  q_conj[..., :3] *= -1.0
  return _quat_apply(q_conj, v)


def _quat_to_body_ang_vel(q_t: np.ndarray, q_tp1: np.ndarray, dt: float) -> np.ndarray:
  """Body-frame angular velocity from successive xyzw quaternions.

  ``ω_b = 2 · log(inv(q_t) · q_{t+1}) / dt`` ≈ ``2 · (Δq.xyz) / dt`` for small Δq.
  Adds shortest-path sign correction so ω is continuous through ±q ambiguity.
  """
  qx, qy, qz, qw = q_t[..., 0], q_t[..., 1], q_t[..., 2], q_t[..., 3]
  px, py, pz, pw = q_tp1[..., 0], q_tp1[..., 1], q_tp1[..., 2], q_tp1[..., 3]
  iqx, iqy, iqz, iqw = -qx, -qy, -qz, qw
  dx = iqw * px + iqx * pw + iqy * pz - iqz * py
  dy = iqw * py - iqx * pz + iqy * pw + iqz * px
  dz = iqw * pz + iqx * py - iqy * px + iqz * pw
  dw = iqw * pw - iqx * px - iqy * py - iqz * pz
  sign = np.where(dw < 0.0, -1.0, 1.0)
  return np.stack([dx * sign, dy * sign, dz * sign], axis=-1) * (2.0 / dt)


def _slerp_arr(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Slerp between two unit quaternion stacks (xyzw); ``t`` shape broadcasts."""
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1_adj = np.where(dot < 0.0, -q1, q1)
  dot = np.abs(dot)
  dot = np.clip(dot, -1.0, 1.0)
  theta = np.arccos(dot)
  sin_t = np.sin(theta)
  small = sin_t < 1e-6
  w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_t))
  w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_t))
  out = w0 * q0 + w1 * q1_adj
  return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _resample_to_fps(
  pos: np.ndarray, quat: np.ndarray, dof: np.ndarray, src_fps: float, dst_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Linear-interp pos/dof and slerp quat to ``dst_fps`` (time-uniform output grid)."""
  if abs(src_fps - dst_fps) < 1e-6:
    return pos, quat, dof
  n_src = pos.shape[0]
  duration = (n_src - 1) / src_fps
  n_dst = max(int(round(duration * dst_fps)) + 1, 2)
  t_dst = np.linspace(0.0, n_src - 1, n_dst)  # source index space
  i0 = np.floor(t_dst).astype(np.int64).clip(0, n_src - 2)
  i1 = i0 + 1
  alpha = (t_dst - i0)[:, None]

  pos_r = pos[i0] * (1.0 - alpha) + pos[i1] * alpha
  dof_r = dof[i0] * (1.0 - alpha) + dof[i1] * alpha
  quat_r = _slerp_arr(quat[i0], quat[i1], alpha)
  return pos_r, quat_r, dof_r


def csv_to_features(
  csv_path: str | Path,
  *,
  src_fps: float = LAFAN1_DEFAULT_FPS,
  target_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Load one LAFAN1-format CSV → ``(features[:-1], features[1:])`` AMP pairs.

  ``target_fps`` triggers linear / slerp resampling so the output (s, s') gap
  matches the policy ``dt``. Pass ``None`` to keep the source rate.
  """
  csv_path = Path(csv_path)
  data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
  if data.ndim != 2 or data.shape[1] != 7 + NUM_DOF_G1:
    raise ValueError(
      f"{csv_path}: expected {7 + NUM_DOF_G1} columns (3 pos + 4 quat + {NUM_DOF_G1} dof), "
      f"got shape {data.shape}"
    )
  pos = data[:, 0:3]
  quat = data[:, 3:7]
  dof = data[:, 7:]
  if target_fps is not None:
    pos, quat, dof = _resample_to_fps(pos, quat, dof, src_fps=src_fps, dst_fps=target_fps)
    fps = target_fps
  else:
    fps = src_fps
  dt = 1.0 / float(fps)

  T = pos.shape[0]
  proj_g = _quat_apply_inverse(quat, np.broadcast_to(GRAVITY_W, (T, 3)).copy())

  lin_vel_w = np.zeros_like(pos)
  lin_vel_w[:-1] = (pos[1:] - pos[:-1]) / dt
  lin_vel_w[-1] = lin_vel_w[-2]
  lin_vel_b = _quat_apply_inverse(quat, lin_vel_w)

  ang_vel_b = np.zeros_like(pos)
  ang_vel_b[:-1] = _quat_to_body_ang_vel(quat[:-1], quat[1:], dt)
  ang_vel_b[-1] = ang_vel_b[-2]

  feats = np.concatenate([proj_g, lin_vel_b, ang_vel_b, dof], axis=-1).astype(np.float32)
  if feats.shape[1] != AMP_FEATURE_DIM:
    raise RuntimeError(
      f"computed feature dim {feats.shape[1]} != expected {AMP_FEATURE_DIM}; "
      f"check {csv_path}"
    )
  return feats[:-1], feats[1:]


class AmpExpertBuffer:
  """Flat (s, s') store for the AMP expert dataset; uniform sampling over pairs.

  The whole dataset lives on ``device`` so ``sample()`` is a single ``torch.randint``
  + index — fast enough to call once per discriminator update.
  """

  def __init__(
    self,
    csv_paths: Iterable[str | Path],
    *,
    device: str | torch.device,
    src_fps: float = LAFAN1_DEFAULT_FPS,
    target_fps: float | None = None,
  ) -> None:
    paths = [Path(p) for p in csv_paths]
    if not paths:
      raise ValueError("AmpExpertBuffer needs at least one CSV path")
    s_chunks: list[np.ndarray] = []
    sp_chunks: list[np.ndarray] = []
    for p in paths:
      s, sp = csv_to_features(p, src_fps=src_fps, target_fps=target_fps)
      s_chunks.append(s)
      sp_chunks.append(sp)
    s_np = np.concatenate(s_chunks, axis=0)
    sp_np = np.concatenate(sp_chunks, axis=0)
    self.s = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)
    self.sp = torch.from_numpy(sp_np).to(device=device, dtype=torch.float32)
    self.feature_dim = int(self.s.shape[-1])
    self.num_pairs = int(self.s.shape[0])
    self.device = self.s.device
    self.csv_paths = paths

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, self.num_pairs, (int(batch_size),), device=self.device)
    return self.s[idx], self.sp[idx]

  def __len__(self) -> int:
    return self.num_pairs
