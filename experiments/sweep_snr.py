#!/usr/bin/env python3
"""Sweep metrics vs SNR (paper Fig. 6). Re-simulates online (no checkpoints).

PyCharm: Run with empty parameters (uses full algo set by default).
Select schemes: Run Configuration → Parameters → ``--algos causal ucb gdo``
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

from sweep_lib import add_common_args, resolved_algos_from_args, sweep_snr
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--snr", type=float, nargs="+",
                   default=[0, 2, 4, 6, 8, 10],
                   help="Target mean best-cell SINR [dB] (paper Fig. 6 axis)")
    args = p.parse_args()
    sweep_snr(algos=resolved_algos_from_args(args), seeds=tuple(args.seeds),
              slots=args.slots, users=args.users, snr_targets=args.snr,
              out_root=args.out_root)
