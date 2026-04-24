"""``otus-play-mjlab`` — play a trained policy in mjlab's native viewer.

Thin wrapper around ``unitree_rl_mjlab``'s ``scripts/play.py``. Use this to
sanity-check a ``.pt`` checkpoint against its original training environment
(MuJoCo-Warp / mjlab), not for the end-goal keyboard sim2sim flow — that one
lives in :mod:`otus_rl_project.deploy.keyboard_play` and runs pure MuJoCo
with an exported ONNX policy.

Typical usage (inside the container)::

    otus-play-mjlab Unitree-G1-Flat \\
        --checkpoint-file runs/rsl_rl/g1_flat/<timestamp>/model_1999.pt
"""

from __future__ import annotations

from otus_rl_project.utils.upstream import run_upstream_script


def main() -> None:
  run_upstream_script("scripts/play.py", argv0="otus-play-mjlab")


if __name__ == "__main__":
  main()
