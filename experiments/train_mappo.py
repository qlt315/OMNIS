#!/usr/bin/env python3
"""Train / evaluate the mappo scheme.

Writes CSVs + series + ``plot_data.mat`` under ``--out`` (default figures/train).
PyCharm: Run with empty parameters. Optional: ``--plot`` to also emit PNGs.
"""
import os
import sys

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from repo_util import ensure_repo_root

ensure_repo_root()

from train_lib import cli_main

if __name__ == "__main__":
    cli_main(default_algos=["mappo"])
