"""Code-block-level scan: per-(MCS, SNR) CB failure rate and residual BER.

Why CB-level: a transport block is K i.i.d. code blocks, so per-branch TB
statistics follow analytically (BLER_TB = 1-(1-p_cb)^K, payload residual BER
~= CB residual BER). Simulating single code blocks instead of full TBs is
6-13x cheaper and makes full-grid scans feasible on a laptop.

Also covers the legacy calibration points: LDPC at rates {0.2, 0.5} with BPSK
over AWGN — the PHY used for the old acc_data.xlsx measurements (rate 1.0 is
uncoded and needs no simulation: BER = Q(sqrt(2*10^(snr/10)))).

Output cb_scan.csv:
    config,mcs_index,qm,code_rate,bg,k,snr_db,cb_fail_rate,residual_ber,n_trials

Smoke:  python run_cb_scan.py --mode mcs --mcs 9 28 --snr 0 10 5 --num-cbs 200
Full:   python run_cb_scan.py --mode mcs --snr -5 20 1   (background, ~1-2 h)
        python run_cb_scan.py --mode legacy
"""

import argparse
import csv
import math
import os
import time
from types import SimpleNamespace

import torch

from link import TBLink, MIN_MOTHER_RATE
from mcs import get_mcs, NUM_MCS

HEADER = ["config", "mcs_index", "qm", "code_rate", "bg", "k",
          "snr_db", "cb_fail_rate", "residual_ber", "n_trials"]

# Reference code-block lengths per base graph (match design_tb's typical picks)
REF_K = {'bg1': 8000, 'bg2': 3693}

LEGACY_RATES = [0.2, 0.5]
# Wide grid: the old sim's coded waterfalls sit ~3-5 dB above NR LDPC's, so we
# must cover both waterfalls to fit the SNR offset delta in the calibration.
LEGACY_SNR = [-5.0 + i for i in range(16)]


def regime_of(code_rate):
    mother = max(code_rate, MIN_MOTHER_RATE)
    return 'bg2' if mother < 1.0 / 3.0 else 'bg1'


def cb_link(qm, code_rate, num_iter, device, precision):
    """Single-CB link at the reference block length of the rate's regime."""
    bg = regime_of(code_rate)
    k = REF_K[bg]
    mcs_shim = SimpleNamespace(qm=qm, code_rate=code_rate)
    return TBLink(k, mcs_shim, num_iter=num_iter, device=device,
                  precision=precision), bg, k


def scan_point(link, snr_db, num_cbs, round_cbs, target_errors, batch_cbs,
               clean_stop=500):
    errs = biterrs = done = 0
    while done < num_cbs and errs < target_errors:
        b = min(round_cbs, num_cbs - done)
        u = torch.randint(0, 2, (b, link.k), dtype=torch.float32)
        u_hat = link.transmit(u, snr_db)
        e = (u_hat != u)
        errs += int((e.sum(dim=1) > 0).sum())
        biterrs += int(e.sum())
        done += b
        # flat region: p_cb ~ 0 needs no more than an upper bound
        if errs == 0 and done >= clean_stop:
            break
    return errs / done, biterrs / (done * link.k), done


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["mcs", "legacy"], default="mcs")
    p.add_argument("--mcs", nargs="+", type=int, default=list(range(NUM_MCS)))
    p.add_argument("--snr", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                   default=[-5.0, 20.0, 1.0])
    p.add_argument("--num-cbs", type=int, default=2000,
                   help="max code blocks per grid point")
    p.add_argument("--round-cbs", type=int, default=250)
    p.add_argument("--target-errors", type=int, default=100)
    p.add_argument("--batch-cbs", type=int, default=250)
    p.add_argument("--num-iter", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--out", default=os.path.join("output", "cb_scan.csv"))
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

        if args.mode == "mcs":
            configs = [("mcs", idx, get_mcs(idx).qm, get_mcs(idx).code_rate,
                        snr_grid(*args.snr)) for idx in args.mcs]
        else:
            configs = [("legacy", -1, 1, r, LEGACY_SNR) for r in LEGACY_RATES]

        for config, mcs_idx, qm, rate, snrs in configs:
            link, bg, k = cb_link(qm, rate, args.num_iter,
                                  args.device, args.precision)
            for snr_db in snrs:
                t0 = time.time()
                p_cb, ber, n = scan_point(link, snr_db, args.num_cbs,
                                          args.round_cbs, args.target_errors,
                                          args.batch_cbs)
                writer.writerow([config, mcs_idx, qm, f"{rate:.4f}", bg, k,
                                 f"{snr_db:.1f}", f"{p_cb:.6f}", f"{ber:.3e}", n])
                f.flush()
                print(f"{config:6s} mcs {mcs_idx:3d} R={rate:.4f} qm={qm} "
                      f"SNR {snr_db:6.1f}: p_cb={p_cb:.4f} ber={ber:.2e} "
                      f"(n={n}) [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
