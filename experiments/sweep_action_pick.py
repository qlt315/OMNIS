#!/usr/bin/env python3
"""Action pick probability (paper Fig. 8) + exploration-knob sweep."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_lib import add_common_args, sweep_action_pick
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--snr", type=float, nargs="+", default=[2, 4, 6])
    p.add_argument("--n-users", type=int, nargs="+", default=[10, 15, 20])
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.55, 1.0, 2.0],
                   help="Exploration knobs: causal_beta / UCB β / mapped DQN ε")
    args = p.parse_args()
    sweep_action_pick(
        algos=args.algos, seeds=tuple(args.seeds), slots=args.slots,
        snr_targets=args.snr, user_list=args.n_users, beta_values=args.betas,
        out_root=args.out_root,
    )
