#!/usr/bin/env python3
"""Deprecated alias for ``convergence_rss.py``."""
from convergence_rss import *  # noqa: F401,F403
from convergence_lib import cli_main

if __name__ == "__main__":
    cli_main(default_algos=["rss"])
