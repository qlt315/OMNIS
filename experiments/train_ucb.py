#!/usr/bin/env python3
"""Deprecated alias for ``convergence_ucb.py``."""
from convergence_ucb import *  # noqa: F401,F403
from convergence_lib import cli_main

if __name__ == "__main__":
    cli_main(default_algos=["ucb"])
