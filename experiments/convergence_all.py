#!/usr/bin/env python3
"""Run online convergence for selected schemes (default: all).

Writes CSVs + series + ``plot_data.mat/.pkl/.npz`` under ``--out``.
Plot PNGs separately with ``plot_results.py`` (or pass ``--plot``).

PyCharm: Run with empty parameters (all algos). To subset, either:
  - set Parameters: ``--algos causal ucb gdo``
  - or edit ``PYCHARM_CONV_ALGOS`` in ``convergence_lib.py``
"""
import os
import sys

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from repo_util import ensure_repo_root

ensure_repo_root()

from convergence_lib import ALGO_NAMES, cli_main

if __name__ == "__main__":
    cli_main(default_algos=list(ALGO_NAMES))
