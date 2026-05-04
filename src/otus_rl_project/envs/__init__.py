"""Custom RL tasks built on top of ``mjlab``'s task registry.

Importing this package registers all otus-specific tasks (e.g. spin kick) so
they become visible to upstream ``unitree_rl_mjlab`` scripts via
``mjlab.tasks.registry``.
"""

from __future__ import annotations

from . import double_kong  # noqa: F401  (import for side-effect: task registration)
from . import locomotion_amp  # noqa: F401
from . import locomotion_compare  # noqa: F401
from . import spinkick  # noqa: F401  (import for side-effect: task registration)
