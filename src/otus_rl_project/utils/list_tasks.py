"""``otus-list-tasks`` — print all tasks registered by mjlab + unitree_rl_mjlab.

The task registry is only populated after the corresponding packages are
imported, so the trick is to trigger both imports and then ask the registry
for the resulting list.
"""

from __future__ import annotations


def main() -> None:
  import mjlab.tasks  # noqa: F401  (populate mjlab side of the registry)
  import src.tasks  # noqa: F401  (populate unitree_rl_mjlab side)
  import otus_rl_project.envs  # noqa: F401  (populate otus side: spinkick, ...)
  from mjlab.tasks.registry import list_tasks

  tasks = sorted(list_tasks())
  if not tasks:
    print("[otus-list-tasks] registry is empty — something is wrong")
    return

  print(f"[otus-list-tasks] {len(tasks)} task(s) available:")
  for t in tasks:
    print(f"  - {t}")


if __name__ == "__main__":
  main()
