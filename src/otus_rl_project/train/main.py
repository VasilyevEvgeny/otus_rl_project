"""Entry point for RL training of the Unitree G1.

This is a placeholder wired into `pyproject.toml` as `otus-train`.
Next milestone: wrap `unitree_rl_mjlab`'s PPO runner so we can just call
`otus-train --task Unitree-G1-Flat --num-envs 4096` and get TensorBoard +
ONNX export.
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro


@dataclass
class TrainArgs:
    task: str = "Unitree-G1-Flat"
    num_envs: int = 4096
    iterations: int = 10_000
    seed: int = 0
    run_name: str | None = None
    wandb_project: str = "otus_rl_project"
    wandb_mode: str = "online"
    headless: bool = True


def main() -> None:
    args = tyro.cli(TrainArgs)
    print(f"[train] placeholder — would start training with args={args}")
    raise NotImplementedError(
        "Training entrypoint is not implemented yet. "
        "Next step: call into unitree_rl_mjlab's task registry and run PPO."
    )


if __name__ == "__main__":
    main()
