"""Sweep (branch, MCS, SNR) -> task accuracy and write acc_table.csv.

Output format (per the PHY data request spec):
    model,mcs_index,snr_db,accuracy,n_trials

Each of the N test payloads is transmitted as one transport block through the
link chain at the given SNR (fresh noise per payload = channel-realization
average), the tail evaluator scores the corrupted reconstructions. Also writes
acc_clean.csv (error-free anchor, one row per branch) on the first branch pass.

Smoke test:
    python run_acc.py --branches Box3 --mcs 9 28 --snr 0 10 5 --num-samples 16
Real run (requires the split-DNN hookup in features.py):
    python run_acc.py --provider npz --num-samples 1000
"""

import argparse
import csv
import os
import time

import torch

from features import (RandomFeatureProvider, SyntheticTailEvaluator,
                      NPZFeatureProvider, NPZTailEvaluator)
from link import TBLink
from mcs import get_mcs, NUM_MCS
from payloads import BRANCHES, payload_bits

HEADER = ["model", "mcs_index", "snr_db", "accuracy", "n_trials"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branches", nargs="+", default=BRANCHES, choices=BRANCHES)
    p.add_argument("--mcs", nargs="+", type=int, default=list(range(NUM_MCS)))
    p.add_argument("--snr", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                   default=[-5.0, 20.0, 1.0])
    p.add_argument("--provider", default="random", choices=["random", "npz"])
    p.add_argument("--num-samples", type=int, default=64,
                   help="test payloads per grid point (production: ~1000)")
    p.add_argument("--batch-samples", type=int, default=16)
    p.add_argument("--num-iter", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--out", default=os.path.join("output", "acc_table.csv"))
    p.add_argument("--clean-out", default=os.path.join("output", "acc_clean.csv"))
    return p.parse_args()


def snr_grid(start, stop, step):
    n = round((stop - start) / step)
    return [start + i * step for i in range(n + 1)]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.provider == "random":
        provider = RandomFeatureProvider(args.num_samples, seed=args.seed)
        evaluator = SyntheticTailEvaluator(provider)
    else:
        provider = NPZFeatureProvider()
        evaluator = NPZTailEvaluator()

    new_file = not os.path.exists(args.out)
    wrote_clean = os.path.exists(args.clean_out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
            f.flush()
        for branch in args.branches:
            payloads = provider.payload_matrix(branch)
            N, P = payloads.shape
            assert P == payload_bits(branch)
            if not wrote_clean:
                acc0 = evaluator.accuracy(branch, payloads)
                with open(args.clean_out, "a", newline="") as fc:
                    w = csv.writer(fc)
                    if os.path.getsize(args.clean_out) == 0:
                        w.writerow(["model", "accuracy"])
                    w.writerow([branch, f"{acc0:.6f}"])
                print(f"{branch}: clean accuracy = {acc0:.4f}")
            for mcs_idx in args.mcs:
                mcs = get_mcs(mcs_idx)
                link = TBLink(P, mcs, num_iter=args.num_iter,
                              device=args.device, precision=args.precision)
                for snr_db in snr_grid(*args.snr):
                    t0 = time.time()
                    corrupted = torch.empty_like(payloads)
                    for i in range(0, N, args.batch_samples):
                        u = payloads[i:i + args.batch_samples]
                        B = u.shape[0]
                        # split P payload bits into K consecutive blocks of k
                        # bits, zero-padding the last block (filler bits)
                        u_pad = torch.zeros(B, link.K * link.k)
                        u_pad[:, :P] = u
                        u_cb = u_pad.reshape(B * link.K, link.k)
                        hat_cb = link.transmit(u_cb, snr_db)
                        corrupted[i:i + B] = hat_cb.reshape(B, link.K * link.k)[:, :P]
                    acc = evaluator.accuracy(branch, corrupted)
                    writer.writerow([branch, mcs_idx, f"{snr_db:.1f}",
                                     f"{acc:.6f}", N])
                    f.flush()
                    print(f"{branch:11s} MCS {mcs_idx:2d} SNR {snr_db:6.1f}: "
                          f"acc={acc:.4f} (N={N}) [{time.time() - t0:.1f}s]")
            wrote_clean = True


if __name__ == "__main__":
    main()
