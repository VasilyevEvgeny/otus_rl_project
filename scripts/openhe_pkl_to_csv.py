"""Convert openhe/g1-retargeted-motions kungfu .pkl to mjlab/LAFAN1 CSV format.

The openhe dataset stores motions retargeted to the 23-DoF Unitree G1
(no wrist joints), as a dict whose only value contains:
  root_trans_offset: (T, 3) — pelvis world position
  root_rot:          (T, 4) — pelvis quaternion in xyzw order
  dof:               (T, 23) — joint angles
  fps:               int

The mjlab tracker (Unitree-G1-Tracking-No-State-Estimation) expects the
LAFAN1 CSV format: 36 columns per frame =
  [x, y, z, qx, qy, qz, qw, dof_0 .. dof_28]
where the 29-DoF order matches Unitree's official G1 description:
  legs (12): L_hip_pitch L_hip_roll L_hip_yaw L_knee L_ankle_pitch L_ankle_roll
             R_hip_pitch R_hip_roll R_hip_yaw R_knee R_ankle_pitch R_ankle_roll
  waist (3): yaw, roll, pitch
  L arm (7): shoulder_pitch shoulder_roll shoulder_yaw elbow wrist_roll wrist_pitch wrist_yaw
  R arm (7): shoulder_pitch shoulder_roll shoulder_yaw elbow wrist_roll wrist_pitch wrist_yaw

The 23-DoF order in openhe drops the 6 wrists (3 per side); padding them
with zeros leaves the wrists in a neutral pose (consistent with the rest
of our LAFAN1-trained policies, which barely use wrists anyway).

Optionally adds smooth start/end transitions from a safe standing pose
(borrowed from mujocolab/g1_spinkick_example) so the tracker has a stable
warm-up and cool-down. This greatly improves training stability for
short, dynamic clips.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SAFE_POSE_29 = np.array(
    [
        -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,
        -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,
         0.0,  0.0,  0.0,
         0.2,  0.2,  0.0,  0.6,  0.0,  0.0,  0.0,
         0.2, -0.2,  0.0,  0.6,  0.0,  0.0,  0.0,
    ],
    dtype=np.float64,
)
SAFE_Z_HEIGHT = 0.76


def pad_23_to_29(dof23: np.ndarray) -> np.ndarray:
    """Map (T, 23) — legs(12) + waist(3) + L arm 4 + R arm 4 — to (T, 29) by
    inserting zeros for the 3+3 wrist DoFs."""
    T = dof23.shape[0]
    dof29 = np.zeros((T, 29), dtype=np.float64)
    dof29[:, 0:12] = dof23[:, 0:12]
    dof29[:, 12:15] = dof23[:, 12:15]
    dof29[:, 15:19] = dof23[:, 15:19]
    dof29[:, 22:26] = dof23[:, 19:23]
    return dof29


def quat_xyzw_to_rotvec(q_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q_xyzw).as_rotvec()


def rotvec_to_quat_xyzw(rv: np.ndarray) -> np.ndarray:
    if np.linalg.norm(rv) < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return Rotation.from_rotvec(rv).as_quat()


def ease_in_cubic(t: float) -> float:
    return t ** 3


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def add_transition(
    pos_a: np.ndarray, rot_a: Rotation, dof_a: np.ndarray,
    pos_b: np.ndarray, rot_b: Rotation, dof_b: np.ndarray,
    n_frames: int, easing,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `n_frames` interpolated frames from (a) -> (b)."""
    pos = np.zeros((n_frames, 3))
    quats = np.zeros((n_frames, 4))
    dofs = np.zeros((n_frames, dof_a.shape[0]))
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_a, rot_b]))
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 1.0
        te = easing(t)
        pos[i]   = pos_a * (1 - te) + pos_b * te
        quats[i] = slerp(te).as_quat()
        dofs[i]  = dof_a * (1 - te) + dof_b * te
    return pos, quats, dofs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Input openhe .pkl")
    ap.add_argument("--csv", required=True, help="Output LAFAN1-format CSV")
    ap.add_argument("--key", default=None,
                    help="Motion key to pick when the pkl is a multi-motion dict "
                         "(e.g. merged_motion.pkl). Substring match is used. "
                         "If None, the only/first motion is used.")
    ap.add_argument("--start-transition", type=float, default=1.0,
                    help="Seconds to ease in from safe standing pose (0 = none).")
    ap.add_argument("--end-transition", type=float, default=1.0,
                    help="Seconds to ease out to safe standing pose (0 = none).")
    ap.add_argument("--end-pad", type=float, default=0.5,
                    help="Seconds of held safe pose after transition (settle).")
    args = ap.parse_args()

    print(f"Loading {args.pkl}...")
    data = joblib.load(args.pkl)
    if not isinstance(data, dict):
        sys.exit("expected dict at top level")
    if args.key:
        matches = [k for k in data if args.key in k]
        if not matches:
            sys.exit(f"no motion key matches {args.key!r}; available: {list(data)}")
        if len(matches) > 1:
            sys.exit(f"ambiguous --key {args.key!r}, matched: {matches}")
        chosen = matches[0]
        motion = data[chosen]
        print(f"  picked motion: {chosen!r}")
    else:
        motion = next(iter(data.values()))
    fps = int(motion["fps"])
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    root_quat = np.asarray(motion["root_rot"], dtype=np.float64)
    dof23 = np.asarray(motion["dof"], dtype=np.float64)
    T = root_pos.shape[0]
    print(f"  {T} frames @ {fps} fps  ({T/fps:.2f}s),  dof23 shape {dof23.shape}")

    dof29 = pad_23_to_29(dof23)

    if args.start_transition > 0.0:
        n = int(args.start_transition * fps)
        target_pos = root_pos[0].copy()
        target_rot = Rotation.from_quat(root_quat[0])
        # Match yaw, drop pitch/roll for the safe start.
        yaw, pitch, roll = target_rot.as_euler("ZYX", degrees=False)
        start_rot = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0])
        start_pos = np.array([target_pos[0], target_pos[1], SAFE_Z_HEIGHT])
        sp, sq, sd = add_transition(
            start_pos, start_rot, SAFE_POSE_29,
            target_pos, target_rot, dof29[0],
            n, ease_in_cubic,
        )
        root_pos  = np.vstack([sp, root_pos])
        root_quat = np.vstack([sq, root_quat])
        dof29     = np.vstack([sd, dof29])
        print(f"  + start transition: {n} frames ({args.start_transition:.1f}s)")

    if args.end_transition > 0.0:
        n = int(args.end_transition * fps)
        start_pos = root_pos[-1].copy()
        start_rot = Rotation.from_quat(root_quat[-1])
        yaw, pitch, roll = start_rot.as_euler("ZYX", degrees=False)
        end_rot = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0])
        end_pos = np.array([start_pos[0], start_pos[1], SAFE_Z_HEIGHT])
        ep, eq, ed = add_transition(
            start_pos, start_rot, dof29[-1],
            end_pos, end_rot, SAFE_POSE_29,
            n, ease_out_cubic,
        )
        root_pos  = np.vstack([root_pos,  ep])
        root_quat = np.vstack([root_quat, eq])
        dof29     = np.vstack([dof29,     ed])
        print(f"  + end transition:   {n} frames ({args.end_transition:.1f}s)")

    if args.end_pad > 0.0:
        n = int(args.end_pad * fps)
        root_pos  = np.vstack([root_pos,  np.tile(root_pos[-1:],  (n, 1))])
        root_quat = np.vstack([root_quat, np.tile(root_quat[-1:], (n, 1))])
        dof29     = np.vstack([dof29,     np.tile(dof29[-1:],     (n, 1))])
        print(f"  + end pad:          {n} frames ({args.end_pad:.1f}s)")

    csv = np.concatenate([root_pos, root_quat, dof29], axis=1)
    print(f"  -> CSV shape {csv.shape}  ({csv.shape[0]/fps:.2f}s @ {fps} fps)")
    np.savetxt(args.csv, csv, delimiter=",", fmt="%.8f")
    print(f"  saved: {args.csv}")


if __name__ == "__main__":
    main()
