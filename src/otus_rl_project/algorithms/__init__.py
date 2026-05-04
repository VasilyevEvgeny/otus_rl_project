"""Roadmap §2 algorithm implementations (ARS, AMP, [stretch] SAC).

Each algorithm lives in its own subpackage. Logging tags are aligned with
``otus_rl_project.eval`` so cross-algorithm plots in TensorBoard / W&B
are trivial — see ``docs/algorithm_comparison_roadmap.md``.

Active subpackages:

- ``ars`` — Augmented Random Search (V2-t)
- ``amp`` — PPO + Adversarial Motion Priors
"""
