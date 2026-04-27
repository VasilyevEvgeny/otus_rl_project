"""``otus-export`` — export a trained rsl_rl checkpoint to ONNX.

Training already saves a ``*.onnx`` next to each ``model_*.pt`` via
``VelocityOnPolicyRunner.save``, so this entry point is mostly for two
follow-up scenarios:

1. You pulled a ``.pt`` checkpoint from elsewhere (e.g. ``wandb`` run
   artifacts) and want a matching ``.onnx`` locally.
2. You are iterating on the export format (different opset, custom metadata)
   without re-running training.

The export pipeline is identical to mjlab's: build the same env+agent config
for the task, instantiate the correct runner class, load the checkpoint,
then call ``runner.export_policy_to_onnx`` and attach base metadata.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

from otus_rl_project.utils.upstream import PROJECT_ROOT, assert_in_container, bootstrap_env


@dataclass
class ExportArgs:
  task: str
  """Registered task ID, e.g. ``Unitree-G1-Flat``. Must match the task used for training."""
  checkpoint: Path
  """Path to a ``model_*.pt`` file produced by ``otus-train``."""
  out: Path | None = None
  """Where to write the ``.onnx``. Defaults to ``<checkpoint_dir>/<checkpoint_stem>.onnx``."""
  device: str = "cpu"
  """Torch device used while instantiating the runner (the final ONNX always runs on CPU weights)."""
  motion_file: Path | None = None
  """For motion-tracking tasks: path to the reference NPZ used during training.
  Required because the env config bakes the motion path; we just need any valid
  motion to instantiate the env so the policy can be loaded."""


def main() -> None:
  assert_in_container()
  bootstrap_env()
  os.chdir(PROJECT_ROOT)

  args = tyro.cli(ExportArgs)
  if not args.checkpoint.exists():
    raise FileNotFoundError(args.checkpoint)

  # Delayed imports so --help stays fast and we don't import torch in the
  # pyproject entry-point resolution path.
  import mjlab.tasks  # noqa: F401  (populate mjlab side of the registry)
  import src.tasks  # noqa: F401  (populate unitree_rl_mjlab side)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)

  env_cfg.scene.num_envs = 1

  if args.motion_file is not None:
    cmds = env_cfg.commands
    motion_cmd = cmds["motion"] if isinstance(cmds, dict) else getattr(cmds, "motion", None)
    if motion_cmd is None or not hasattr(motion_cmd, "motion_file"):
      raise SystemExit(
        f"--motion-file passed but task {args.task!r} has no commands.motion.motion_file"
      )
    motion_cmd.motion_file = str(args.motion_file)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )

  out_path = args.out or args.checkpoint.with_suffix(".onnx")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  # export_policy_to_onnx takes a directory + filename, so split accordingly.
  runner.export_policy_to_onnx(str(out_path.parent), out_path.name)

  metadata = get_base_metadata(env.unwrapped, run_path="local-export")
  attach_metadata_to_onnx(str(out_path), metadata)

  env.close()
  print(f"[otus-export] wrote {out_path}")


if __name__ == "__main__":
  main()
