"""Detect kicks-with-spin in a motion NPZ.

A 'kick with spin' = a time window where, simultaneously:
  - one ankle is elevated above the pelvis (high foot)
  - pelvis yaw rate exceeds ~180 deg/s

We also report 'high foot' events alone (any kick) and 'fast spin' events alone
(any pirouette), so it's clear what kind of motion is in the clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quat_wxyz_to_yaw(quat: np.ndarray) -> np.ndarray:
    """Extract yaw angle from a wxyz quaternion array of shape (T, 4)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def find_body_idx(npz: dict, candidates: list[str]) -> int | None:
    """Try to locate a body index. NPZ format may not include names; we use
    structural assumptions: pelvis is the root body (index 0), feet are the
    bodies with the lowest mean z.
    """
    body_z = npz["body_pos_w"][:, :, 2].mean(axis=0)  # (B,)
    if "ankle" in candidates[0] or "foot" in candidates[0]:
        # two lowest-z bodies are the feet
        order = np.argsort(body_z)
        return int(order[0])  # caller will pull two lowest separately
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--high-foot-frac", type=float, default=0.0,
                    help="A foot is 'high' if its z is at least this fraction "
                         "of pelvis z above the pelvis. 0.0 = above pelvis.")
    ap.add_argument("--spin-rate-dps", type=float, default=180.0,
                    help="Pelvis yaw rate threshold for 'fast spin' (deg/s).")
    args = ap.parse_args()

    d = np.load(args.npz)
    body_pos = d["body_pos_w"]      # (T, B, 3)
    body_quat = d["body_quat_w"]    # (T, B, 4) wxyz
    fps = float(d["fps"][0])
    T, B, _ = body_pos.shape
    print(f"motion: {args.npz.name}")
    print(f"  duration: {T/fps:.1f}s  ({T} frames @ {fps:.0f} fps, {B} bodies)")

    # Heuristic body indices.
    # Mean-z over the whole clip identifies feet (lowest) and pelvis (low-but-not-foot).
    mean_z = body_pos[:, :, 2].mean(axis=0)
    feet_idx = list(np.argsort(mean_z)[:2])   # 2 lowest = left/right ankle
    pelvis_idx = 0  # body 0 is the root for mjlab/G1

    print(f"  pelvis idx: {pelvis_idx}  (mean z {mean_z[pelvis_idx]:.2f}m)")
    print(f"  feet idx:   {feet_idx}    (mean z {mean_z[feet_idx[0]]:.2f}m, "
          f"{mean_z[feet_idx[1]]:.2f}m)")

    pelvis_z = body_pos[:, pelvis_idx, 2]                 # (T,)
    foot_z = body_pos[:, feet_idx, 2]                     # (T, 2)

    # 1) high-foot mask: any foot above pelvis (extra threshold = fraction of pelvis_z)
    foot_above_pelvis = foot_z - pelvis_z[:, None]        # (T, 2)
    high_foot = foot_above_pelvis.max(axis=1) > (args.high_foot_frac * pelvis_z)
    high_idx = np.where(high_foot)[0]

    # 2) yaw rate of pelvis
    pelvis_yaw = quat_wxyz_to_yaw(body_quat[:, pelvis_idx])
    pelvis_yaw_unwrapped = np.unwrap(pelvis_yaw)
    yaw_rate_dps = np.gradient(pelvis_yaw_unwrapped, 1.0 / fps) * 180.0 / np.pi
    fast_spin = np.abs(yaw_rate_dps) > args.spin_rate_dps
    spin_idx = np.where(fast_spin)[0]

    # 3) "kick with spin" = a high-foot event where the average pelvis yaw
    #    rate within +/- 0.6 s exceeds 100 deg/s. Real roundhouse kicks have
    #    the spin and the kick separated in time by a few hundred ms, so a
    #    strict "same-frame" intersection is too tight.
    win_frames = max(1, int(0.6 * fps))
    yaw_rate_abs = np.abs(yaw_rate_dps)
    yaw_rate_smooth = np.convolve(
        yaw_rate_abs,
        np.ones(2 * win_frames + 1) / (2 * win_frames + 1),
        mode="same",
    )
    kick_with_spin = high_foot & (yaw_rate_smooth > 100.0)
    ks_idx = np.where(kick_with_spin)[0]

    def merge_into_events(idx: np.ndarray, fps: float, gap_s: float = 0.2) -> list[tuple[float, float]]:
        """Group consecutive frames (with gaps <gap_s) into events."""
        if len(idx) == 0:
            return []
        gap_frames = max(1, int(gap_s * fps))
        events = []
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k - prev > gap_frames:
                events.append((start / fps, prev / fps))
                start = k
            prev = k
        events.append((start / fps, prev / fps))
        return events

    # report
    def report(name: str, events: list[tuple[float, float]], extra=""):
        print(f"\n{name} ({len(events)} events){extra}")
        for s, e in events[:20]:
            print(f"  {s:6.2f}s  -  {e:6.2f}s   (dur {e - s:.2f}s)")
        if len(events) > 20:
            print(f"  ... and {len(events) - 20} more")

    print(f"\nMax foot height above pelvis: {foot_above_pelvis.max():+.2f}m")
    print(f"Max |pelvis yaw rate|:         {np.abs(yaw_rate_dps).max():.0f} deg/s")

    report("HIGH FOOT  (foot above pelvis level — any kick)",
           merge_into_events(high_idx, fps))
    report(f"FAST SPIN  (>{args.spin_rate_dps:.0f} deg/s yaw rate)",
           merge_into_events(spin_idx, fps))
    report("KICK WITH SPIN  (high foot + sustained yaw within ±0.6s)",
           merge_into_events(ks_idx, fps),
           extra=f"  threshold: foot above pelvis & avg|yaw_rate|>100 deg/s nearby")

    # Also report ALL kicks at a softer threshold: foot above knee height (~0.5 m).
    knee_z = 0.5
    soft_kick = (foot_z.max(axis=1) > knee_z) & (foot_z.max(axis=1) > pelvis_z * 0.5)
    soft_idx = np.where(soft_kick)[0]
    report(f"KNEE-HIGH FOOT  (a foot above {knee_z:.1f}m, softer 'any kick')",
           merge_into_events(soft_idx, fps))


if __name__ == "__main__":
    main()
