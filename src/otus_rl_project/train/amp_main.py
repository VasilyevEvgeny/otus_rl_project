"""``otus-train-amp`` — PPO + AMP on the comparison locomotion task.

Thin wrapper around upstream ``unitree_rl_mjlab/scripts/train.py``: we register
``Otus-G1-Walk-AMP`` in :mod:`otus_rl_project.envs` (custom runner =
:class:`~otus_rl_project.algorithms.amp.amp_runner.AmpVelocityRunner`) and
auto-inject that task as the first positional argument, so the user can keep
the familiar ``--agent.*`` / ``--env.*`` tyro flags.

Required CLI:

    otus-train-amp --agent.amp.expert-csv-paths src/assets/motions/g1/walk1.csv

Common knobs::

    --agent.max-iterations 5000
    --agent.amp.amp-reward-weight 0.5
    --agent.amp.amp-replace-style-terms True
    --agent.amp.disc-batch-size 4096
    --agent.locomotion-eval-interval 50
"""

from __future__ import annotations

import sys

from otus_rl_project.utils.upstream import run_upstream_script

TASK_ID = "Otus-G1-Walk-AMP"


def main() -> None:
  if not any(arg == TASK_ID for arg in sys.argv[1:]):
    sys.argv.insert(1, TASK_ID)
  run_upstream_script("scripts/train.py", argv0="otus-train-amp")


if __name__ == "__main__":
  main()
