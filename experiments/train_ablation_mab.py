#!/usr/bin/env python3
"""Train MAB ablations (base / no_update / freeze) + Causal prediction-error series.

Writes under ``--out`` (default ``figures/ablation``) so overnight sweeps are
not clobbered. Plot with ``experiments/plot_ablation.py``.

Usage:
  PYTHONPATH=. python3 experiments/train_ablation_mab.py \\
      --slots 300 --users 10 --seeds 0 1 2 --freeze-after 50 \\
      --out figures/ablation
  PYTHONPATH=. python3 experiments/train_ablation_mab.py --bases causal ucb
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import MAB_BASE_ALGOS, run_mab_ablations  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="MAB freeze / no-update ablations + Causal pred-error logging")
    p.add_argument(
        "--bases", nargs="+", default=list(MAB_BASE_ALGOS),
        metavar="NAME",
        help=f"MAB bases to ablate (default: all). Choices: {', '.join(MAB_BASE_ALGOS)}")
    p.add_argument("--slots", type=int, default=300)
    p.add_argument("--users", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--freeze-after", type=int, default=50,
                   help="slots with GP updates before freeze variant stops learning")
    p.add_argument("--out", default="figures/ablation")
    args = p.parse_args()
    run_mab_ablations(
        bases=args.bases, slots=args.slots, users=args.users,
        seeds=tuple(args.seeds), freeze_after=args.freeze_after, out=args.out)


if __name__ == "__main__":
    main()
