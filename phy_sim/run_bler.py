"""Sweep (branch, MCS, SNR) -> TB-level BLER and write bler_table.csv.

Output format (per the PHY data request spec):
    model,mcs_index,snr_db,bler,n_trials

Adaptive trials: each grid point runs in rounds of --round-tbs transport
blocks and stops once --target-errors block errors are observed (tight
statistics near the waterfall) or --num-tbs is reached (flat regions).
Rows are appended incrementally so interrupted runs keep their progress.

Examples
--------
Smoke test (minutes on CPU):
    python run_bler.py --branches Box3 --mcs 9 16 28 --snr -2 12 2 --num-tbs 200
Production (GPU server):
    python run_bler.py --num-tbs 50000 --target-errors 200 --device cuda
"""

import argparse
import csv
import os
import time

import torch

from link import TBLink
from mcs import get_mcs, NUM_MCS
from payloads import BRANCHES, payload_bits

HEADER = ["model", "mcs_index", "snr_db", "bler", "n_trials"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branches", nargs="+", default=BRANCHES, choices=BRANCHES)
    p.add_argument("--mcs", nargs="+", type=int, default=list(range(NUM_MCS)))
    p.add_argument("--snr", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                   default=[-5.0, 20.0, 1.0])
    p.add_argument("--num-tbs", type=int, default=20000,
                   help="max transport blocks per grid point")
    p.add_argument("--round-tbs", type=int, default=1000,
                   help="transport blocks per adaptive round")
    p.add_argument("--target-errors", type=int, default=100,
                   help="stop a grid point after this many block errors")
    p.add_argument("--batch-tbs", type=int, default=8,
                   help="TBs simulated in parallel (memory vs speed)")
    p.add_argument("--num-iter", type=int, default=30, help="LDPC decoder iterations")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=os.path.join("output", "bler_table.csv"))
    return p.parse_args()


def snr_grid(start, stop, step):
    n = round((stop - start) / step)
    return [start + i * step for i in range(n + 1)]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
            f.flush()
        for branch in args.branches:
            for mcs_idx in args.mcs:
                mcs = get_mcs(mcs_idx)
                link = TBLink(payload_bits(branch), mcs, num_iter=args.num_iter,
                              device=args.device, precision=args.precision)
                for snr_db in snr_grid(*args.snr):
                    t0 = time.time()
                    tb_err = n_done = 0
                    while n_done < args.num_tbs and tb_err < args.target_errors:
                        b = min(args.round_tbs, args.num_tbs - n_done)
                        r = link.simulate_tb(b, snr_db, batch_tbs=args.batch_tbs)
                        tb_err += r["tb_errors"]
                        n_done += r["num_tbs"]
                    bler = tb_err / n_done
                    writer.writerow([branch, mcs_idx, f"{snr_db:.1f}",
                                     f"{bler:.6f}", n_done])
                    f.flush()
                    print(f"{branch:11s} MCS {mcs_idx:2d} SNR {snr_db:6.1f}: "
                          f"BLER={bler:.4f} ({tb_err}/{n_done}) "
                          f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
