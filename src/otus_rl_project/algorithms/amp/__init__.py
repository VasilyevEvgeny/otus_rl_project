"""AMP (Adversarial Motion Priors, Peng et al., 2021) for G1 locomotion.

Plugs an LSGAN-style state-pair discriminator on top of the regular velocity
PPO baseline: the discriminator learns to tell expert (LAFAN1) (s, s') pairs
apart from policy ones, and the resulting score is folded into the env reward
as a "style" term. See ``docs/algorithm_comparison_roadmap.md`` §2 / phase 2.
"""

from __future__ import annotations

from otus_rl_project.algorithms.amp.discriminator import AmpDiscriminator
from otus_rl_project.algorithms.amp.expert_buffer import (
  AmpExpertBuffer,
  csv_to_features,
)

__all__ = [
  "AmpDiscriminator",
  "AmpExpertBuffer",
  "csv_to_features",
]
