"""Trim a tracking-format NPZ motion file to a frame range.

Usage:
  python trim_motion.py --input motion.npz --output trimmed.npz \
      --start-frame 80 --end-frame 260
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--input", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--start-frame", type=int, required=True)
  p.add_argument("--end-frame", type=int, required=True, help="exclusive")
  args = p.parse_args()

  src = np.load(args.input)
  s, e = args.start_frame, args.end_frame

  out = {}
  for key in src.files:
    arr = src[key]
    if arr.ndim == 0:
      out[key] = arr
    elif arr.shape[0] >= e:
      out[key] = arr[s:e]
    else:
      out[key] = arr
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.output, **out)
  fps_arr = np.asarray(out["fps"]).reshape(-1)
  fps = float(fps_arr[0]) if fps_arr.size else 30.0
  T = out["body_pos_w"].shape[0]
  print(f"OK -> {args.output}  frames={T}  duration={T / fps:.2f}s @ {fps} fps")


if __name__ == "__main__":
  main()
