#!/usr/bin/env python3
"""Sweep metrics vs number of MDs (paper Fig. 7). Re-simulates online."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_lib import add_common_args, sweep_users
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--n-users", type=int, nargs="+",
                   default=[5, 10, 15, 20, 25],
                   help="MD counts (nested prefixes of 25-UE pool)")
    p.add_argument("--snr-db", type=float, default=5.0,
                   help="Fixed target mean SINR [dB] during user sweep")
    args = p.parse_args()
    sweep_users(algos=args.algos, seeds=tuple(args.seeds), slots=args.slots,
                user_list=args.n_users, snr_db=args.snr_db,
                out_root=args.out_root)
