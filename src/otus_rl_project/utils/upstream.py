"""Helpers for delegating CLI entry points to upstream ``unitree_rl_mjlab``.

``unitree_rl_mjlab`` is cloned into ``/opt/third_party/unitree_rl_mjlab`` at
image build time. Its ``scripts/{train,play,list_envs}.py`` files use a
project-specific ``tyro`` CLI that we do not want to duplicate; instead we run
them via :func:`runpy.run_path` so our ``otus-*`` commands behave identically
to the upstream scripts with the same arguments.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

UNITREE_RL_MJLAB_ROOT = Path("/opt/third_party/unitree_rl_mjlab")
PROJECT_ROOT = Path("/workspace/otus_rl_project")


def assert_in_container() -> None:
  if not UNITREE_RL_MJLAB_ROOT.exists():
    raise FileNotFoundError(
      f"Upstream unitree_rl_mjlab not found at {UNITREE_RL_MJLAB_ROOT}. "
      "These commands are designed to run inside the otus_rl_project Docker "
      "container (see `make up && make attach`)."
    )


def bootstrap_env() -> None:
  """Apply sane defaults for headless MuJoCo + wandb/tensorboard dirs."""
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("WANDB_DIR", str(PROJECT_ROOT / "runs" / "wandb"))
  os.environ.setdefault("TENSORBOARD_DIR", str(PROJECT_ROOT / "runs" / "tb"))
  (PROJECT_ROOT / "runs" / "wandb").mkdir(parents=True, exist_ok=True)
  (PROJECT_ROOT / "runs" / "tb").mkdir(parents=True, exist_ok=True)


def run_upstream_script(relative_path: str, argv0: str) -> None:
  """Delegate to an upstream script as if it were invoked directly.

  Args:
    relative_path: Path under ``unitree_rl_mjlab/`` (e.g. ``scripts/train.py``).
    argv0: Value to put in ``sys.argv[0]`` so help text mentions our name.
  """
  assert_in_container()
  bootstrap_env()

  # logs/ tree is relative to CWD in upstream — pin it to our repo.
  os.chdir(PROJECT_ROOT)
  sys.argv[0] = argv0
  runpy.run_path(str(UNITREE_RL_MJLAB_ROOT / relative_path), run_name="__main__")
