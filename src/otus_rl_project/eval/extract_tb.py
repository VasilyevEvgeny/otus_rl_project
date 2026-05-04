"""``otus-extract-tb`` — extract Eval/AMP scalar tags from TensorBoard logs into CSV + figures + summary table.

Phase-4 helper for the algorithm-comparison study (see ``docs/algorithm_comparison_roadmap.md``).
Given one or more TB run directories (each labeled by algorithm), it reads the scalar event
files, exports per-run CSV, generates one PNG per metric overlaying every run, and writes a
markdown table of final-window means used directly by ``docs/results/comparison_report.md``.

Typical usage (inside the container — note the **single** ``--run`` flag with multiple values)::

    otus-extract-tb \
        --run ppo=logs/rsl_rl/g1_walk_compare/<run_dir> \
              amp=logs/rsl_rl/g1_walk_amp/<run_dir> \
              ars=logs/rsl_rl/g1_walk_compare_ars/<run_dir> \
        --out-dir docs/results

Or, from the host, ``make compare-extract`` auto-discovers the most recent run of each
algorithm under ``logs/rsl_rl/`` and runs the extractor inside the container.

All outputs are deterministic given the same input event files, so re-running after a fresh
training overwrites the figures + table in place.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import tyro

# Headless backend; must be set before any pyplot import.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


@dataclass
class ExtractArgs:
  run: list[str] = field(default_factory=list)
  """One or more ``label=path`` pairs, e.g. ``--run ppo=logs/.../run_dir``. Order is preserved."""
  out_dir: Path = Path("docs/results")
  """Root output directory. Subdirs ``csv/``, ``figures/``, ``tables/`` are created."""
  tag_prefixes: tuple[str, ...] = ("Eval/", "AMP/")
  """Only scalar tags starting with one of these prefixes are extracted."""
  figure_prefixes: tuple[str, ...] = ("Eval/",)
  """Only tags with these prefixes get an overlaid PNG figure (Eval/* by design — same tag set across runs)."""
  final_window_frac: float = 0.1
  """Fraction of the **trailing** scalar samples averaged for the summary table (0.1 = last 10%)."""
  summary_metrics: tuple[str, ...] = (
    "Eval/velocity_tracking_error_xy",
    "Eval/yaw_tracking_error",
    "Eval/episode_length",
    "Eval/base_height_std",
    "Eval/torque_cost",
    "Eval/action_smoothness",
    "Eval/foot_slippage",
    "Eval/velocity_tracking_error_xy_obs_noise",
    "Eval/velocity_tracking_error_xy_push_5N",
    "Eval/velocity_tracking_error_xy_push_15N",
  )
  """Headline metrics for the comparison table. Missing metrics are rendered as ``-``."""


def _parse_runs(run_args: list[str]) -> list[tuple[str, Path]]:
  if not run_args:
    raise SystemExit("--run is required (at least one label=path pair)")
  parsed: list[tuple[str, Path]] = []
  for arg in run_args:
    if "=" not in arg:
      raise SystemExit(f"--run expects label=path, got {arg!r}")
    label, _, path = arg.partition("=")
    label = label.strip()
    p = Path(path).expanduser()
    if not p.exists():
      raise FileNotFoundError(f"run dir not found: {p}")
    if not p.is_dir():
      raise NotADirectoryError(f"--run path must be a directory: {p}")
    parsed.append((label, p))
  return parsed


def _safe_tag(tag: str) -> str:
  return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")


def _load_scalars(
  run_dir: Path, prefixes: tuple[str, ...]
) -> dict[str, list[tuple[int, float]]]:
  """Aggregate scalar events from every TF event file under ``run_dir``."""
  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  scalars: dict[str, list[tuple[int, float]]] = {}
  for tag in ea.Tags().get("scalars", []):
    if not any(tag.startswith(p) for p in prefixes):
      continue
    series = [(int(ev.step), float(ev.value)) for ev in ea.Scalars(tag)]
    series.sort(key=lambda x: x[0])
    scalars[tag] = series
  return scalars


def _write_csv(path: Path, series: list[tuple[int, float]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "value"])
    writer.writerows(series)


def _final_window_mean(series: list[tuple[int, float]], frac: float) -> float | None:
  if not series:
    return None
  n = len(series)
  k = max(1, int(round(n * frac)))
  values = [v for _, v in series[-k:]]
  return sum(values) / len(values)


def _plot_metric(
  tag: str, run_data: list[tuple[str, list[tuple[int, float]]]], out_path: Path
) -> None:
  """Overlay one curve per run on a single figure for ``tag``."""
  fig, ax = plt.subplots(figsize=(7.0, 4.2))
  any_data = False
  for label, series in run_data:
    if not series:
      continue
    xs = [s for s, _ in series]
    ys = [v for _, v in series]
    ax.plot(xs, ys, label=label, linewidth=1.6)
    any_data = True
  if not any_data:
    plt.close(fig)
    return
  ax.set_xlabel("iteration")
  ax.set_ylabel(tag)
  ax.set_title(tag)
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", frameon=False)
  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=130)
  plt.close(fig)


def _format_value(metric: str, val: float | None) -> str:
  if val is None:
    return "—"
  if "episode_length" in metric:
    return f"{val:.0f}"
  if abs(val) >= 100:
    return f"{val:.1f}"
  if abs(val) >= 1:
    return f"{val:.3f}"
  return f"{val:.4f}"


def _write_summary_table(
  table_path: Path,
  run_labels: list[str],
  per_run_finals: dict[str, dict[str, float | None]],
  metrics: tuple[str, ...],
  final_frac: float,
) -> None:
  """Render a markdown comparison table; bold the best (lowest-error / longest-episode) per row."""
  table_path.parent.mkdir(parents=True, exist_ok=True)

  better_higher = {"Eval/episode_length"}

  lines: list[str] = []
  lines.append("# Cross-algorithm Eval/* summary")
  lines.append("")
  lines.append(
    f"_Final-window mean (last {final_frac:.0%} of TB samples) per run, per metric._"
  )
  lines.append("")
  header = "| Metric | " + " | ".join(run_labels) + " |"
  sep = "|---" * (1 + len(run_labels)) + "|"
  lines.append(header)
  lines.append(sep)

  for metric in metrics:
    row_vals = {label: per_run_finals.get(label, {}).get(metric) for label in run_labels}
    valid = [(lbl, v) for lbl, v in row_vals.items() if v is not None]
    best_label: str | None = None
    if valid:
      if metric in better_higher:
        best_label = max(valid, key=lambda kv: kv[1])[0]
      else:
        best_label = min(valid, key=lambda kv: kv[1])[0]
    cells = []
    for label in run_labels:
      val = row_vals[label]
      formatted = _format_value(metric, val)
      if label == best_label and val is not None:
        formatted = f"**{formatted}**"
      cells.append(formatted)
    lines.append("| `" + metric + "` | " + " | ".join(cells) + " |")

  lines.append("")
  lines.append(
    "Bold = best run on that metric "
    "(lower is better, except `Eval/episode_length` where higher is better)."
  )
  lines.append("")
  table_path.write_text("\n".join(lines))


def main() -> None:
  args = tyro.cli(ExtractArgs)
  runs = _parse_runs(args.run)
  out_dir = args.out_dir.expanduser().resolve()
  csv_dir = out_dir / "csv"
  fig_dir = out_dir / "figures"
  tab_dir = out_dir / "tables"

  print(f"[extract-tb] runs ({len(runs)}):")
  for label, path in runs:
    print(f"  {label}: {path}")
  print(f"[extract-tb] output dir: {out_dir}")

  per_run_scalars: dict[str, dict[str, list[tuple[int, float]]]] = {}
  for label, run_dir in runs:
    scalars = _load_scalars(run_dir, args.tag_prefixes)
    per_run_scalars[label] = scalars
    print(f"[extract-tb] {label}: {len(scalars)} matching scalar tags")
    for tag, series in scalars.items():
      _write_csv(csv_dir / f"{label}__{_safe_tag(tag)}.csv", series)

  all_tags: set[str] = set()
  for scalars in per_run_scalars.values():
    all_tags.update(scalars.keys())
  fig_tags = sorted(t for t in all_tags if any(t.startswith(p) for p in args.figure_prefixes))
  print(f"[extract-tb] generating {len(fig_tags)} comparison figures")

  for tag in fig_tags:
    run_data = [(label, per_run_scalars[label].get(tag, [])) for label, _ in runs]
    _plot_metric(tag, run_data, fig_dir / f"{_safe_tag(tag)}.png")

  per_run_finals: dict[str, dict[str, float | None]] = {}
  for label, scalars in per_run_scalars.items():
    per_run_finals[label] = {
      tag: _final_window_mean(series, args.final_window_frac)
      for tag, series in scalars.items()
    }

  run_labels = [label for label, _ in runs]
  _write_summary_table(
    tab_dir / "eval_summary.md",
    run_labels,
    per_run_finals,
    args.summary_metrics,
    args.final_window_frac,
  )

  print(f"[extract-tb] wrote csv -> {csv_dir}")
  print(f"[extract-tb] wrote figures -> {fig_dir}")
  print(f"[extract-tb] wrote summary table -> {tab_dir / 'eval_summary.md'}")


if __name__ == "__main__":
  main()
