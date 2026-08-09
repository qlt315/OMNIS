#!/usr/bin/env python3
"""Sweep metrics vs number of MDs (paper Fig. 7). Re-simulates online.

PyCharm: Run with empty parameters (users 5–25, all algos by default).
Select schemes: ``--algos causal ucb gdo`` or ``--algos all``.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

from sweep_lib import add_common_args, resolved_algos_from_args, sweep_users
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
    sweep_users(algos=resolved_algos_from_args(args), seeds=tuple(args.seeds),
                slots=args.slots, user_list=args.n_users, snr_db=args.snr_db,
                out_root=args.out_root)
