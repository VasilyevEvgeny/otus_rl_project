"""Export a trained rsl_rl checkpoint (.pt) to ONNX for keyboard sim2sim play.

Placeholder. rsl_rl ships with an ONNX exporter in `rsl_rl.modules`; the plan
is to load the checkpoint, grab the actor (deterministic), and save with
constant observation/action shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class ExportArgs:
    checkpoint: Path
    out: Path
    opset: int = 17


def main() -> None:
    args = tyro.cli(ExportArgs)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] placeholder — would export {args.checkpoint} -> {args.out}")
    raise NotImplementedError(
        "ONNX export is not implemented yet. "
        "Next step: load rsl_rl ActorCritic, extract actor MLP, torch.onnx.export."
    )


if __name__ == "__main__":
    main()
