"""Soft Actor-Critic (SAC) implementation for the locomotion compare task.

Off-policy alternative to PPO/AMP/ARS in roadmap §2 (stretch goal). Uses the
same :class:`Otus-G1-Walk-Compare` env config and the shared eval-callback
(:mod:`otus_rl_project.eval`) so cross-algorithm metrics are directly comparable.
"""
