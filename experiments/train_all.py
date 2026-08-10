#!/usr/bin/env python3
"""Deprecated alias for ``convergence_all.py``."""
from convergence_all import *  # noqa: F401,F403
from convergence_lib import ALGO_NAMES, cli_main

if __name__ == "__main__":
    cli_main(default_algos=list(ALGO_NAMES))
