#!/usr/bin/env python3
"""Train / evaluate selected schemes (default: all).

Writes/merges CSVs + series under --out. Does **not** plot.
Plot separately with experiments/plot_results.py.

Examples:
  PYTHONPATH=. python3 experiments/train_all.py
  PYTHONPATH=. python3 experiments/train_all.py --algos causal ucb gdo dqn
  PYTHONPATH=. python3 experiments/train_all.py --algos cto --seeds 0 1 2 3 4
  PYTHONPATH=. python3 experiments/plot_results.py --indir figures
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import ALGO_NAMES, cli_main

if __name__ == "__main__":
    cli_main(default_algos=list(ALGO_NAMES))
