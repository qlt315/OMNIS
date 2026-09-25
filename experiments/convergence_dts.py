#!/usr/bin/env python3
"""Online convergence for dts."""
import os
import sys

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from repo_util import ensure_repo_root

ensure_repo_root()

from convergence_lib import cli_main

if __name__ == "__main__":
    cli_main(default_algos=["dts"])
