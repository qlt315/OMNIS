#!/usr/bin/env python3
"""Sweep metrics vs Poisson task arrival rate (journal queueing extension)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_lib import add_common_args, sweep_arrival
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.30, 0.50, 0.70, 0.90, 1.10],
                   help="Uniform per-user arrival rate [tasks/slot]")
    p.add_argument("--snr-db", type=float, default=5.0)
    args = p.parse_args()
    sweep_arrival(algos=args.algos, seeds=tuple(args.seeds), slots=args.slots,
                  users=args.users, rates=args.rates, snr_db=args.snr_db,
                  out_root=args.out_root)
