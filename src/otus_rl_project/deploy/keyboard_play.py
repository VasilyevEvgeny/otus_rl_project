"""Sim2sim deploy: run an ONNX policy in the MuJoCo passive viewer with keyboard control.

Controls (once implemented):
    W / S      — forward / backward linear velocity command
    A / D      — left / right lateral velocity command
    Q / E      — yaw rate (turn left / right)
    SPACE      — reset to nominal pose
    ESC        — quit

Placeholder. The final version will:
    1. Load MJCF from assets/g1/scene_mjx.xml
    2. Load policy.onnx via onnxruntime (CPU is fine for inference at 50 Hz)
    3. Build the observation exactly as training did (joint pos/vel, base
       orientation, angular velocity, velocity command, last action)
    4. PD-controlled step_ctrl = nominal + action_scale * action
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class PlayArgs:
    policy: Path
    mjcf: Path = Path("/workspace/otus_rl_project/assets/g1/scene_mjx.xml")
    action_scale: float = 0.25
    control_hz: int = 50


def main() -> None:
    args = tyro.cli(PlayArgs)
    if not args.policy.exists():
        raise FileNotFoundError(args.policy)
    if not args.mjcf.exists():
        raise FileNotFoundError(
            f"MJCF not found at {args.mjcf}. Run `make assets` first."
        )
    print(f"[play] placeholder — would play {args.policy} on {args.mjcf}")
    raise NotImplementedError(
        "Keyboard sim2sim play is not implemented yet. "
        "Next step: integrate mujoco.viewer.launch_passive + onnxruntime."
    )


if __name__ == "__main__":
    main()
