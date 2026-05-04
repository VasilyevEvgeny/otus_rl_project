"""Augmented Random Search (Mania et al., 2018) for Unitree G1 locomotion.

Implements ARS-V2 (state-normalized, top-:math:`b` filtering) on top of
:class:`mjlab.rl.RslRlVecEnvWrapper` so it shares the same task / scene
config as the PPO baseline. See ``docs/algorithm_comparison_roadmap.md`` §2.
"""

from __future__ import annotations

from otus_rl_project.algorithms.ars.policy import (
  ArsActorAdapter,
  LinearPolicy,
  RunningMeanStd,
)
from otus_rl_project.algorithms.ars.runner import ArsRunner, ArsRunnerCfg

__all__ = [
  "ArsActorAdapter",
  "ArsRunner",
  "ArsRunnerCfg",
  "LinearPolicy",
  "RunningMeanStd",
]
