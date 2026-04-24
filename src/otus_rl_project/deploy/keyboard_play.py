"""``otus-play`` — sim2sim keyboard deploy.

Runs an **ONNX policy** (exported via ``otus-export`` or emitted automatically
during training) inside a **plain MuJoCo** passive viewer (not mjlab /
MuJoCo-Warp). This is the actual sim-to-sim check: if the policy survives
here, it has a reasonable chance of surviving sim-to-real on hardware.

The policy's 7 metadata fields embedded by ``attach_metadata_to_onnx`` are
enough to rebuild the runtime without any external configuration:

* ``joint_names``       — actuator order in the training env;
* ``default_joint_pos`` — nominal joint pose (origin of ``joint_pos_rel``);
* ``joint_stiffness``   — per-joint kp (overrides MJCF defaults);
* ``joint_damping``     — per-joint kd (overrides MJCF defaults);
* ``action_scale``      — per-joint scaling from raw action to delta-angle;
* ``observation_names`` — ordered list of obs terms;
* ``command_names``     — list of command names the policy expects.

The actor observation for velocity-tracking tasks (default Unitree-G1-Flat)
is the concatenation:

    [base_ang_vel(3) | projected_gravity(3) | command(3) |
     phase(2)        | joint_pos_rel(N)     | joint_vel(N) | last_action(N)]

with ``phase`` zeroed when ``‖command‖ < 0.1`` and a period of ``0.6`` s,
matching ``unitree_rl_mjlab/src/tasks/velocity/mdp/observations.py``.

Keyboard (inside MuJoCo viewer window):

    W / S   -> vx += / -= lin_vel_step
    A / D   -> vy += / -= lin_vel_step
    Q / E   -> wz += / -= ang_vel_step
    SPACE   -> zero command (stand still)
    R       -> reset robot to nominal pose
    ESC     -> close viewer (use window X button)

Keys are one-shot (MuJoCo passive viewer doesn't expose held-key state),
so each press bumps the command by a step — this is deliberate and makes
benchmarking easier.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

log = logging.getLogger("otus-play")


@dataclass
class PlayArgs:
  policy: Path
  """Path to a ``.onnx`` file produced by ``otus-export`` or mjlab training."""
  mjcf: Path = Path("/workspace/otus_rl_project/assets/g1/scene.xml")
  """MJCF scene used by plain MuJoCo. Default is the G1 standing scene."""
  control_hz: int = 50
  """Policy inference rate (Hz). Physics substeps are computed from MJCF timestep."""
  lin_vel_step: float = 0.1
  """Increment added to vx/vy per key press."""
  lin_vel_max: float = 1.0
  """Saturation limit for vx/vy."""
  ang_vel_step: float = 0.2
  """Increment added to yaw_rate per key press."""
  ang_vel_max: float = 1.0
  """Saturation limit for yaw_rate."""
  initial_height: float = 0.793
  """Base z at reset (m). Roughly G1 nominal standing height."""
  phase_period_s: float = 0.6
  """Gait clock period, matches velocity_env_cfg.py ``phase.period``."""


_GLFW_W = 87
_GLFW_A = 65
_GLFW_S = 83
_GLFW_D = 68
_GLFW_Q = 81
_GLFW_E = 69
_GLFW_R = 82
_GLFW_SPACE = 32


def _parse_csv_floats(s: str) -> np.ndarray:
  return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def _load_onnx_metadata(onnx_path: Path) -> dict:
  import onnx

  model = onnx.load(str(onnx_path))
  meta = {p.key: p.value for p in model.metadata_props}
  required = [
    "joint_names",
    "default_joint_pos",
    "joint_stiffness",
    "joint_damping",
    "action_scale",
    "observation_names",
    "command_names",
  ]
  missing = [k for k in required if k not in meta]
  if missing:
    raise RuntimeError(
      f"ONNX {onnx_path} is missing metadata fields: {missing}. "
      "Re-export with `otus-export` (mjlab embeds them via attach_metadata_to_onnx)."
    )
  return {
    "joint_names": meta["joint_names"].split(","),
    "default_joint_pos": _parse_csv_floats(meta["default_joint_pos"]),
    "joint_stiffness": _parse_csv_floats(meta["joint_stiffness"]),
    "joint_damping": _parse_csv_floats(meta["joint_damping"]),
    "action_scale": _parse_csv_floats(meta["action_scale"]),
    "observation_names": meta["observation_names"].split(","),
    "command_names": meta["command_names"].split(","),
  }


def _build_indices(mj_model, joint_names: list[str]):
  import mujoco

  qpos_inds, qvel_inds, ctrl_inds = [], [], []
  for jn in joint_names:
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if jid < 0:
      raise RuntimeError(f"Joint '{jn}' not found in MJCF")
    qpos_inds.append(mj_model.jnt_qposadr[jid])
    qvel_inds.append(mj_model.jnt_dofadr[jid])

    aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
    if aid < 0:
      raise RuntimeError(f"Actuator '{jn}' not found in MJCF")
    ctrl_inds.append(aid)
  return (
    np.asarray(qpos_inds, dtype=np.int32),
    np.asarray(qvel_inds, dtype=np.int32),
    np.asarray(ctrl_inds, dtype=np.int32),
  )


def _override_pd_gains(mj_model, ctrl_inds: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
  """Make MJCF position actuators use the training-time PD gains."""
  # MuJoCo position actuator semantics (per mj docs):
  #   force = gainprm[0] * (qpos_target - qpos) + biasprm[1] * qpos + biasprm[2] * qvel
  # Training convention is  force = kp * (target - qpos) - kd * qvel, which maps to:
  #   gainprm[0] =  kp
  #   biasprm[0] =  0
  #   biasprm[1] = -kp
  #   biasprm[2] = -kd
  for cid, kp_i, kd_i in zip(ctrl_inds, kp, kd, strict=True):
    mj_model.actuator_gainprm[cid, 0] = kp_i
    mj_model.actuator_biasprm[cid, 0] = 0.0
    mj_model.actuator_biasprm[cid, 1] = -kp_i
    mj_model.actuator_biasprm[cid, 2] = -kd_i


class _CommandState:
  """Thread-safe holder for twist command + reset request flag."""

  def __init__(self, args: PlayArgs) -> None:
    self._args = args
    self._lock = threading.Lock()
    self._cmd = np.zeros(3, dtype=np.float32)
    self._reset_requested = False

  def snapshot(self) -> np.ndarray:
    with self._lock:
      return self._cmd.copy()

  def pop_reset(self) -> bool:
    with self._lock:
      r, self._reset_requested = self._reset_requested, False
      return r

  def handle_key(self, keycode: int) -> None:
    a = self._args
    with self._lock:
      if keycode == _GLFW_W:
        self._cmd[0] = np.clip(self._cmd[0] + a.lin_vel_step, -a.lin_vel_max, a.lin_vel_max)
      elif keycode == _GLFW_S:
        self._cmd[0] = np.clip(self._cmd[0] - a.lin_vel_step, -a.lin_vel_max, a.lin_vel_max)
      elif keycode == _GLFW_A:
        self._cmd[1] = np.clip(self._cmd[1] + a.lin_vel_step, -a.lin_vel_max, a.lin_vel_max)
      elif keycode == _GLFW_D:
        self._cmd[1] = np.clip(self._cmd[1] - a.lin_vel_step, -a.lin_vel_max, a.lin_vel_max)
      elif keycode == _GLFW_Q:
        self._cmd[2] = np.clip(self._cmd[2] + a.ang_vel_step, -a.ang_vel_max, a.ang_vel_max)
      elif keycode == _GLFW_E:
        self._cmd[2] = np.clip(self._cmd[2] - a.ang_vel_step, -a.ang_vel_max, a.ang_vel_max)
      elif keycode == _GLFW_SPACE:
        self._cmd[:] = 0.0
      elif keycode == _GLFW_R:
        self._reset_requested = True
        self._cmd[:] = 0.0
      else:
        return
    log.info(
      "cmd: vx=%+.2f vy=%+.2f wz=%+.2f", self._cmd[0], self._cmd[1], self._cmd[2]
    )


def _reset_state(
  mj_model,
  mj_data,
  qpos_inds: np.ndarray,
  default_qpos: np.ndarray,
  initial_height: float,
) -> None:
  import mujoco

  mujoco.mj_resetData(mj_model, mj_data)
  # Free-joint root (MuJoCo convention: qpos[0:3] pos, qpos[3:7] quat [w,x,y,z]).
  mj_data.qpos[0:3] = [0.0, 0.0, initial_height]
  mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  for idx, q0 in zip(qpos_inds, default_qpos, strict=True):
    mj_data.qpos[idx] = float(q0)
  mj_data.qvel[:] = 0.0
  mujoco.mj_forward(mj_model, mj_data)


def _build_observation(
  mj_model,
  mj_data,
  qpos_inds: np.ndarray,
  qvel_inds: np.ndarray,
  default_qpos: np.ndarray,
  command: np.ndarray,
  last_action: np.ndarray,
  step_counter: int,
  control_dt: float,
  phase_period_s: float,
) -> np.ndarray:
  import mujoco

  # Rotation world->body from base quaternion (mujoco quat: [w,x,y,z]).
  quat = mj_data.qpos[3:7].astype(np.float64)
  rot = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rot, quat)
  rot = rot.reshape(3, 3)
  rot_T = rot.T

  # IMU-style angular velocity: world -> body.
  ang_vel_world = mj_data.qvel[3:6].astype(np.float64)
  base_ang_vel = rot_T @ ang_vel_world

  # Projected gravity (unit vector; mjlab uses unit gravity, not ±9.81).
  proj_grav = rot_T @ np.array([0.0, 0.0, -1.0])

  # Phase: zero if command is tiny, else sin/cos on 0.6 s clock.
  if float(np.linalg.norm(command)) < 0.1:
    phase = np.zeros(2, dtype=np.float32)
  else:
    t = step_counter * control_dt
    p = (t % phase_period_s) / phase_period_s * 2.0 * np.pi
    phase = np.array([np.sin(p), np.cos(p)], dtype=np.float32)

  joint_pos_rel = mj_data.qpos[qpos_inds].astype(np.float32) - default_qpos
  joint_vel = mj_data.qvel[qvel_inds].astype(np.float32)

  obs = np.concatenate(
    [
      base_ang_vel.astype(np.float32),
      proj_grav.astype(np.float32),
      command.astype(np.float32),
      phase,
      joint_pos_rel,
      joint_vel,
      last_action.astype(np.float32),
    ]
  )[None, :]  # shape [1, obs_dim]
  return obs


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
  args = tyro.cli(PlayArgs)
  if not args.policy.exists():
    raise FileNotFoundError(args.policy)
  if not args.mjcf.exists():
    raise FileNotFoundError(
      f"MJCF not found at {args.mjcf}. Run `make assets` first."
    )

  # Late imports so --help stays fast.
  import mujoco
  import mujoco.viewer
  import onnxruntime as ort

  meta = _load_onnx_metadata(args.policy)
  joint_names = meta["joint_names"]
  default_qpos = meta["default_joint_pos"]
  action_scale = meta["action_scale"]
  kp = meta["joint_stiffness"]
  kd = meta["joint_damping"]
  expected_obs_terms = meta["observation_names"]
  log.info("policy: %s", args.policy)
  log.info("joints (%d): %s", len(joint_names), joint_names)
  log.info("observation_names: %s", expected_obs_terms)
  log.info("command_names: %s", meta["command_names"])

  mj_model = mujoco.MjModel.from_xml_path(str(args.mjcf))
  mj_data = mujoco.MjData(mj_model)
  qpos_inds, qvel_inds, ctrl_inds = _build_indices(mj_model, joint_names)
  _override_pd_gains(mj_model, ctrl_inds, kp, kd)
  _reset_state(mj_model, mj_data, qpos_inds, default_qpos, args.initial_height)

  sess = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
  inp_name = sess.get_inputs()[0].name
  out_name = sess.get_outputs()[0].name
  onnx_obs_dim = sess.get_inputs()[0].shape[-1]
  onnx_act_dim = sess.get_outputs()[0].shape[-1]
  if onnx_act_dim != len(joint_names):
    raise RuntimeError(
      f"ONNX action dim {onnx_act_dim} != number of joints {len(joint_names)}"
    )

  cmd_state = _CommandState(args)

  # Sanity-check observation dim by building one with zero command.
  test_obs = _build_observation(
    mj_model,
    mj_data,
    qpos_inds,
    qvel_inds,
    default_qpos,
    cmd_state.snapshot(),
    np.zeros(len(joint_names), dtype=np.float32),
    0,
    1.0 / args.control_hz,
    args.phase_period_s,
  )
  if test_obs.shape[-1] != onnx_obs_dim:
    raise RuntimeError(
      f"observation dim {test_obs.shape[-1]} != ONNX expected {onnx_obs_dim}. "
      f"Observation terms from metadata: {expected_obs_terms}. "
      "Only velocity-task obs layout is supported right now."
    )
  log.info("obs dim: %d, act dim: %d — shapes match ONNX", onnx_obs_dim, onnx_act_dim)

  control_dt = 1.0 / args.control_hz
  sim_steps = max(1, int(round(control_dt / mj_model.opt.timestep)))
  log.info(
    "control_hz=%d, MJCF timestep=%.4fs -> %d physics steps per policy step",
    args.control_hz,
    mj_model.opt.timestep,
    sim_steps,
  )

  last_action = np.zeros(len(joint_names), dtype=np.float32)
  step_counter = 0

  print(
    "\nControls:\n"
    "  W/S A/D  — vx / vy   (step / press)\n"
    "  Q/E      — yaw rate\n"
    "  SPACE    — zero command\n"
    "  R        — reset robot\n"
  )

  with mujoco.viewer.launch_passive(
    mj_model, mj_data, key_callback=cmd_state.handle_key
  ) as viewer:
    while viewer.is_running():
      loop_start = time.time()

      if cmd_state.pop_reset():
        _reset_state(mj_model, mj_data, qpos_inds, default_qpos, args.initial_height)
        last_action[:] = 0
        step_counter = 0

      command = cmd_state.snapshot()
      obs = _build_observation(
        mj_model,
        mj_data,
        qpos_inds,
        qvel_inds,
        default_qpos,
        command,
        last_action,
        step_counter,
        control_dt,
        args.phase_period_s,
      )
      action = sess.run([out_name], {inp_name: obs})[0][0]
      action = np.clip(action, -10.0, 10.0)
      last_action = action.astype(np.float32)

      target_q = default_qpos + action_scale * action
      mj_data.ctrl[ctrl_inds] = target_q

      for _ in range(sim_steps):
        mujoco.mj_step(mj_model, mj_data)

      step_counter += 1
      viewer.sync()

      elapsed = time.time() - loop_start
      sleep_for = control_dt - elapsed
      if sleep_for > 0:
        time.sleep(sleep_for)


if __name__ == "__main__":
  main()
