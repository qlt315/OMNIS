#!/usr/bin/env python3
"""Deprecated alias for ``convergence_dts.py``."""
from convergence_dts import *  # noqa: F401,F403
from convergence_lib import cli_main

if __name__ == "__main__":
    cli_main(default_algos=["dts"])
