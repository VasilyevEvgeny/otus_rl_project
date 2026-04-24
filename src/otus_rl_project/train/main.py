"""``otus-train`` — RL training entry point for the Unitree G1.

Thin wrapper around ``unitree_rl_mjlab``'s ``scripts/train.py`` (cloned into
``/opt/third_party/unitree_rl_mjlab`` at image build time). The upstream
script owns the ``tyro`` CLI; we only prepare the environment and delegate.

Typical usage (inside the container)::

    otus-train Unitree-G1-Flat --env.scene.num-envs=4096
    otus-train Unitree-G1-Flat --agent.max-iterations=2000 --gpu-ids 0

The checkpoint directory ``logs/rsl_rl/<experiment>/<timestamp>/`` is created
under ``/workspace/otus_rl_project/``, which is bind-mounted from the host.
Each checkpoint save also writes a ``*.onnx`` next to ``model_*.pt`` thanks
to ``VelocityOnPolicyRunner.save`` in mjlab.
"""

from __future__ import annotations

from otus_rl_project.utils.upstream import run_upstream_script


def main() -> None:
  run_upstream_script("scripts/train.py", argv0="otus-train")


if __name__ == "__main__":
  main()
