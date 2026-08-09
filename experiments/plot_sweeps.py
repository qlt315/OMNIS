#!/usr/bin/env python3
"""Re-plot sweep results and refresh ``*.mat`` from ``perseed.csv``.

Does **not** re-run simulations. Use after ``run_sweeps.py`` / ``sweep_*.py``.

PyCharm: edit ``PYCHARM_*`` below, Run with empty parameters.

CLI:
  PYTHONPATH=. python3 experiments/plot_sweeps.py
  PYTHONPATH=. python3 experiments/plot_sweeps.py --only snr users
  PYTHONPATH=. python3 experiments/plot_sweeps.py --indir figures/sweeps/snr
"""
from __future__ import annotations

import argparse
import os

from repo_util import ensure_repo_root

ensure_repo_root()

from sweep_lib import (  # noqa: E402
    replot_action_pick_dir,
    replot_sweep_dir,
    resolve_algos,
)

# =============================================================================
# PyCharm defaults
# =============================================================================
PYCHARM_ROOT = "figures/sweeps"
# Empty → all subdirs that contain perseed.csv / perseed_*.csv
PYCHARM_ONLY = []  # e.g. ["snr", "users", "arrival", "action_pick"]
PYCHARM_ALGOS = None  # None = whatever is in CSV; or ["causal","ucb"] / "all"
# =============================================================================

SWEEP_NAMES = ("snr", "users", "arrival", "action_pick")


def _has_sweep_csv(d):
    if os.path.isfile(os.path.join(d, "perseed.csv")):
        return True
    if not os.path.isdir(d):
        return False
    for name in os.listdir(d):
        if name.startswith("perseed") and name.endswith(".csv"):
            return True
    return False


def _discover_sweep_dirs(root, only):
    root = os.path.abspath(root)
    if _has_sweep_csv(root):
        return [root]
    names = list(only) if only else list(SWEEP_NAMES)
    dirs = []
    for name in names:
        d = os.path.join(root, name)
        if _has_sweep_csv(d):
            dirs.append(d)
    if not only and os.path.isdir(root):
        for ent in sorted(os.listdir(root)):
            d = os.path.join(root, ent)
            if os.path.isdir(d) and _has_sweep_csv(d) and d not in dirs:
                dirs.append(d)
    return dirs


def _replot_one(d, algos):
    name = os.path.basename(d.rstrip(os.sep))
    print(f"=== replot {name} ({d}) ===", flush=True)
    if name == "action_pick" or (
            not os.path.isfile(os.path.join(d, "perseed.csv"))
            and _has_sweep_csv(d)):
        # Multi-CSV action_pick layout (or only perseed_*.csv present)
        if name == "action_pick" or any(
                n.startswith("perseed_") and n.endswith(".csv")
                for n in os.listdir(d)):
            return replot_action_pick_dir(d, algos=algos)
    return replot_sweep_dir(d, sweep_name=name, algos=algos)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=None,
                   help="sweeps root (default: figures/sweeps)")
    p.add_argument("--indir", default=None,
                   help="single sweep dir with perseed.csv (overrides --root/--only)")
    p.add_argument("--only", nargs="+", default=None,
                   choices=list(SWEEP_NAMES),
                   help="subset of sweeps under --root")
    p.add_argument("--algos", nargs="+", default=None,
                   help="subset of algorithms to plot; default = all in CSV")
    args = p.parse_args(argv)

    root = args.root if args.root is not None else PYCHARM_ROOT
    only = args.only if args.only is not None else list(PYCHARM_ONLY or [])
    algos = None
    if args.algos is not None:
        algos = resolve_algos(args.algos)
    elif PYCHARM_ALGOS is not None:
        algos = resolve_algos(
            PYCHARM_ALGOS if PYCHARM_ALGOS != "all" else ["all"])

    if args.indir:
        dirs = [args.indir]
    else:
        dirs = _discover_sweep_dirs(root, only)

    if not dirs:
        raise SystemExit(
            f"no sweep perseed.csv found under {root!r}; "
            "run experiments/run_sweeps.py first")

    for d in dirs:
        _replot_one(d, algos)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
