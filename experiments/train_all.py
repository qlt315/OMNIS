#!/usr/bin/env python3
"""Train / evaluate all schemes. Writes figures/ CSVs and plots."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import ALGOS, cli_main

if __name__ == "__main__":
    cli_main(default_algos=[a for a, _ in ALGOS])
