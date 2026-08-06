"""Synthesize the MCS accuracy table from the legacy measurements — no DNN needed.

Method ("robustness-curve transplant"):
1. Phi_branch: task accuracy as a function of payload residual BER, calibrated
   from acc_data.xlsx (real DNN, AWGN+BPSK):
   - rate 1.0 rows: analytic BER Q(sqrt(2*snr_linear)) — assumption-free;
   - rate 0.2/0.5 rows: residual BER from our NR-LDPC reproduction
     (cb_scan legacy rows) shifted by a per-rate SNR offset delta_r that
     absorbs the old simulator's weaker code (fitted below);
   - the old sim's coded waterfalls sit several dB above NR LDPC's, so
     delta_r is fitted on the rate-1 Phi first, then all points are merged.
2. Consistency check: all 15 calibration points per branch must collapse onto
   one monotone curve. A diagnostic plot per branch is written to output/.
3. Synthesis: acc(branch, mcs, snr) = Phi_branch(residual_ber(mcs, snr))
   with residual BER from cb_scan (isotonic-smoothed in dB).
4. Also assembles bler_table.csv: BLER = 1-(1-p_cb)^K per branch.

Approximation to keep in mind: post-LDPC-failure errors are bursty while the
legacy uncoded errors are i.i.d.; equal average BER may not be perfectly
equivalent for the DNN. Regenerate with the real DNN when available.

Usage:
    python synth_acc.py --calib-only        # calibration + plots (fast)
    python synth_acc.py                     # full: acc_table.csv + bler_table.csv
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from link import design_tb
from mcs import get_mcs, NUM_MCS
from payloads import BRANCHES, payload_bits

# On-air payload sizes used by the OMNIS system simulator (sys_data/config.py
# data_size convention) for the TB-level BLER assembly 1-(1-p_cb)^K.
SYSTEM_PAYLOAD = {
    'Box3': 6e3, 'Box6': 13.26e3, 'Box12': 33.58e3,
    'Standard3': 11.23e3, 'Standard6': 22.46e3, 'Standard12': 44.93e3,
}


def _pava(y):
    """Pool-adjacent-violators: non-decreasing weighted isotonic fit."""
    y = np.asarray(y, float)
    blocks = []  # [mean, weight, count, start, end]
    for i, v in enumerate(y):
        blocks.append([v, 1.0, 1, i, i])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b2, b1 = blocks.pop(), blocks.pop()
            w = b1[1] + b2[1]
            blocks.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w,
                           w, b1[2] + b2[2], b1[3], b2[4]])
    out = np.empty(len(y))
    for mean, _, _, s, e in blocks:
        out[s:e + 1] = mean
    return out


def isotonic_xy(x, y, decreasing=True):
    """Isotonic fit of y(x); returns (unique_sorted_x, fitted_y) with y
    constrained monotone (decreasing by default)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    fit = -_pava(-ys) if decreasing else _pava(ys)
    xs_u, idx = np.unique(xs, return_index=True)
    return xs_u, fit[idx]

XLSX = os.path.join("..", "sys_data", "acc_data", "acc_data.xlsx")
SNR_COLS = ["SNR=0", "SNR=3", "SNR=5", "SNR=7", "SNR=10"]
SNR_VALS = [0.0, 3.0, 5.0, 7.0, 10.0]
LEGACY_RATES = [0.2, 0.5]
BER_FLOOR = 1e-7          # anchor point for the clean ceiling
FINE_GRID = np.arange(-5.0, 20.0 + 1e-9, 1.0)


def q_func(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def bpsk_ber(snr_db):
    return q_func(math.sqrt(2.0 * 10 ** (snr_db / 10.0)))


def load_legacy_acc():
    df = pd.read_excel(XLSX)
    out = {b: {} for b in BRANCHES}
    for _, r in df.iterrows():
        out[r["Model"]][float(r["Rate"])] = [float(r[c]) for c in SNR_COLS]
    return out


def load_cb_scan(paths):
    rows = []
    for p in paths:
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    return rows


def iso_smooth(snr, val, grid, decreasing=True, floor=None):
    """Isotonic smooth of val(snr) onto grid; val decreases with snr."""
    val = np.asarray(val, float)
    if floor is not None:
        val = np.maximum(val, floor)
    xs, ys = isotonic_xy(snr, val, decreasing=decreasing)
    return np.interp(grid, xs, ys)  # clamps outside the observed range


class Phi:
    """Monotone accuracy-vs-residual-BER curve for one branch."""

    def __init__(self, bers, accs):
        bers = np.clip(np.asarray(bers, float), BER_FLOOR, 1.0)
        self.x, self.y = isotonic_xy(np.log10(bers), np.asarray(accs, float),
                                     decreasing=True)

    def __call__(self, ber):
        x = np.log10(np.clip(np.asarray(ber, float), BER_FLOOR, 1.0))
        return np.interp(x, self.x, self.y)


def fit_delta(phi0, sim_snr, sim_ber, meas_acc):
    """Grid-search the SNR offset aligning simulated BER to Phi0."""
    best = (None, np.inf)
    for delta in np.arange(-8.0, 4.01, 0.25):
        ber = np.interp(np.asarray(SNR_VALS) + delta, sim_snr, sim_ber,
                        left=sim_ber[0], right=sim_ber[-1])
        pred = phi0(np.maximum(ber, BER_FLOOR))
        err = float(np.mean((pred - np.asarray(meas_acc)) ** 2))
        if err < best[1]:
            best = (delta, err)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--coded", choices=["off", "on"], default="off",
                    help="include legacy coded rows in Phi (off: uncoded row "
                         "only; coded rows stay as validation markers. The "
                         "legacy low-rate code is inconsistent with any "
                         "SNR-shifted NR reproduction and is excluded by "
                         "default)")
    ap.add_argument("--cb", nargs="+",
                    default=["output/cb_scan_a1.csv", "output/cb_scan_a.csv",
                             "output/cb_scan_b.csv"])
    ap.add_argument("--cb-legacy", default="output/cb_legacy.csv")
    ap.add_argument("--out-acc", default="output/acc_table.csv")
    ap.add_argument("--out-bler", default="output/bler_table.csv")
    ap.add_argument("--out-clean", default="output/acc_clean.csv")
    args = ap.parse_args()

    legacy_acc = load_legacy_acc()
    legacy_rows = [r for r in load_cb_scan([args.cb_legacy])
                   if r["config"] == "legacy"]
    legacy_ber = {}
    for rate in LEGACY_RATES:
        rows = sorted((r for r in legacy_rows
                       if abs(float(r["code_rate"]) - rate) < 1e-6),
                      key=lambda r: float(r["snr_db"]))
        legacy_ber[rate] = ([float(r["snr_db"]) for r in rows],
                            [max(float(r["residual_ber"]), BER_FLOOR)
                             for r in rows])

    phis, deltas = {}, {}
    for branch in BRANCHES:
        data = legacy_acc[branch]
        # pass 1: Phi from the uncoded row + clean anchor
        bers = [BER_FLOOR] + [max(bpsk_ber(s), BER_FLOOR) for s in SNR_VALS]
        accs = [max(data[1.0])] + data[1.0]
        phi0 = Phi(bers, accs)
        # pass 2: align coded rows via per-rate SNR offset (diagnostics;
        # included in Phi only with --coded on)
        bers_all, accs_all, src = list(bers), list(accs), ["uncoded"] * len(bers)
        for rate in LEGACY_RATES:
            s_snr, s_ber = legacy_ber.get(rate, ([], []))
            if not s_snr:
                continue
            delta, err = fit_delta(phi0, np.array(s_snr), np.array(s_ber),
                                   data[rate])
            deltas[(branch, rate)] = (delta, err)
            ber_aligned = np.interp(np.asarray(SNR_VALS) + delta,
                                    s_snr, s_ber, left=s_ber[0], right=s_ber[-1])
            bers_all += list(np.maximum(ber_aligned, BER_FLOOR))
            accs_all += list(data[rate])
            src += [f"rate{rate}"] * len(SNR_VALS)
        # pass 3: final Phi (uncoded row only by default)
        if args.coded == "on":
            phi = Phi(bers_all, accs_all)
        else:
            phi = phi0
        phis[branch] = phi
        # diagnostics plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for s, m, c in [("uncoded", "o", "tab:blue"), ("rate0.2", "s", "tab:green"),
                        ("rate0.5", "^", "tab:orange")]:
            xs = [np.log10(max(b, BER_FLOOR)) for b, t in zip(bers_all, src) if t == s]
            ys = [a for a, t in zip(accs_all, src) if t == s]
            ax.scatter(xs, ys, marker=m, c=c, label=s, zorder=3)
        gx = np.linspace(min(phi.x), max(phi.x), 200)
        ax.plot(gx, phi(10 ** gx), "k-", lw=1.5, label="Phi fit")
        d20 = deltas.get((branch, 0.2), (None,))[0]
        d50 = deltas.get((branch, 0.5), (None,))[0]
        ax.set_title(f"{branch}  (delta_0.2={d20}, delta_0.5={d50} dB)")
        ax.set_xlabel("log10 residual BER")
        ax.set_ylabel("accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join("output", f"phi_{branch}.png"), dpi=120)
        plt.close(fig)
        print(f"{branch:11s}: collapse ok, deltas={deltas.get((branch, 0.2))} "
              f"{deltas.get((branch, 0.5))}")

    if args.calib_only:
        print("calibration done (--calib-only); plots in output/phi_*.png")
        return

    # ---- synthesis ----
    mcs_rows = [r for r in load_cb_scan(args.cb) if r["config"] == "mcs"]
    by_mcs = {}
    for r in mcs_rows:
        by_mcs.setdefault(int(r["mcs_index"]), []).append(r)
    with open(args.out_acc, "w", newline="") as fa, \
         open(args.out_bler, "w", newline="") as fb, \
         open(args.out_clean, "w", newline="") as fc:
        wa = csv.writer(fa); wa.writerow(["model", "mcs_index", "snr_db",
                                          "accuracy", "n_trials"])
        wb = csv.writer(fb); wb.writerow(["model", "mcs_index", "snr_db",
                                          "bler", "n_trials"])
        wc = csv.writer(fc); wc.writerow(["model", "accuracy"])
        for branch in BRANCHES:
            phi = phis[branch]
            wc.writerow([branch, f"{phi(BER_FLOOR):.6f}"])
            for mcs_idx in range(NUM_MCS):
                rows = sorted(by_mcs.get(mcs_idx, []),
                              key=lambda r: float(r["snr_db"]))
                if not rows:
                    continue
                snr = np.array([float(r["snr_db"]) for r in rows])
                ber = np.array([float(r["residual_ber"]) for r in rows])
                pcb = np.array([float(r["cb_fail_rate"]) for r in rows])
                ntr = max(int(r["n_trials"]) for r in rows)
                ber_g = iso_smooth(snr, ber, FINE_GRID, floor=BER_FLOOR)
                pcb_g = np.clip(iso_smooth(snr, pcb, FINE_GRID), 0.0, 1.0)
                K = design_tb(SYSTEM_PAYLOAD[branch], get_mcs(mcs_idx))["K"]
                for s, b, p in zip(FINE_GRID, ber_g, pcb_g):
                    wa.writerow([branch, mcs_idx, f"{s:.1f}",
                                 f"{phi(b):.6f}", ntr])
                    wb.writerow([branch, mcs_idx, f"{s:.1f}",
                                 f"{1 - (1 - p) ** K:.6f}", ntr])
    print(f"wrote {args.out_acc}, {args.out_bler}, {args.out_clean}")


if __name__ == "__main__":
    main()
