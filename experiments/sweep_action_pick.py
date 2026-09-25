#!/usr/bin/env python3
"""Action pick probability vs SNR / #MDs."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

from sweep_lib import add_common_args, resolved_algos_from_args, sweep_action_pick
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--snr", type=float, nargs="+", default=[2, 4, 6])
    p.add_argument("--n-users", type=int, nargs="+", default=[10, 15, 20])
    args = p.parse_args()
    sweep_action_pick(
        algos=resolved_algos_from_args(args), seeds=tuple(args.seeds),
        slots=args.slots, snr_targets=args.snr, user_list=args.n_users,
        out_root=args.out_root,
    )
