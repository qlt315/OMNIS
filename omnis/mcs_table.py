import csv
import os

import numpy as np


class McsTable:
    """Lookup layer for the Sionna-generated PHY tables.

    Tables (AWGN-conditional, indexed by per-slot SINR in dB):
        mcs_def.csv    : mcs_index -> (modulation, qm, code_rate, spectral_efficiency)
        bler_table.csv : (model, mcs_index, snr_db) -> transport-block error rate
        acc_clean.csv  : model -> error-free accuracy ceiling (COCO mAP)

    Accuracy semantics (BLER-gated, matching the single-cell Sionna design):
        E[acc] = (1 - BLER) * clean_mAP + BLER * floor_mAP
    A failed TB is an erasure (floor, default 0), not a residual-BER corruption.
    Latency / queue service use goodput SE = (1 - BLER) * spectral_efficiency.
    """

    def __init__(self, table_dir, acc_floor=0.0, bler_target=0.1):
        self.table_dir = table_dir
        self.acc_floor = float(acc_floor)
        self.bler_target = float(bler_target)
        self.se = {}            # mcs_index -> spectral efficiency [bit/s/Hz]
        self.qm = {}
        self.code_rate = {}
        self.modulation = {}
        with open(os.path.join(table_dir, "mcs_def.csv")) as f:
            for r in csv.DictReader(f):
                idx = int(r["mcs_index"])
                self.modulation[idx] = r["modulation"]
                self.qm[idx] = int(r["qm"])
                self.code_rate[idx] = float(r["code_rate"])
                self.se[idx] = float(r["spectral_efficiency"])
        self.num_mcs = len(self.se)
        self.mcs_indices = sorted(self.se.keys())

        self.acc_clean = {}
        with open(os.path.join(table_dir, "acc_clean.csv")) as f:
            for r in csv.DictReader(f):
                self.acc_clean[r["model"]] = float(r["accuracy"])

        self._bler = self._load_curves("bler_table.csv", "bler")

    def _load_curves(self, filename, value_col):
        curves = {}
        with open(os.path.join(self.table_dir, filename)) as f:
            for r in csv.DictReader(f):
                model = r["model"]
                mcs = int(r["mcs_index"])
                curves.setdefault(model, {}).setdefault(mcs, []).append(
                    (float(r["snr_db"]), float(r[value_col])))
        for model in curves:
            for mcs in curves[model]:
                pts = sorted(curves[model][mcs])
                curves[model][mcs] = (
                    np.array([p[0] for p in pts]),
                    np.array([p[1] for p in pts]),
                )
        return curves

    def _interp(self, curves, model, mcs_idx, snr_db):
        snr, val = curves[model][mcs_idx]
        return float(np.interp(snr_db, snr, val))  # clamps outside the grid

    def bler(self, model, mcs_idx, snr_db):
        """Interpolated transport-block error rate."""
        return float(np.clip(self._interp(self._bler, model, mcs_idx, snr_db), 0.0, 1.0))

    def accuracy(self, model, mcs_idx, snr_db):
        """BLER-gated expected task accuracy (COCO mAP).

        Failed TBs are treated as erasures at ``acc_floor`` (default 0); successful
        TBs deliver the clean mAP. Matches the classmate Sionna B2 semantics.
        """
        clean = self.acc_clean[model]
        p_fail = self.bler(model, mcs_idx, snr_db)
        return (1.0 - p_fail) * clean + p_fail * self.acc_floor

    def goodput_se(self, model, mcs_idx, snr_db):
        """Effective spectral efficiency after TB erasures [bit/s/Hz]."""
        return self.se[mcs_idx] * (1.0 - self.bler(model, mcs_idx, snr_db))

    def best_mcs_by_acc(self, model, snr_db):
        """MCS with the highest BLER-gated accuracy (ties -> higher raw SE)."""
        best, best_key = 0, (-np.inf, -np.inf)
        for mcs in self.mcs_indices:
            key = (self.accuracy(model, mcs, snr_db), self.se[mcs])
            if key > best_key:
                best, best_key = mcs, key
        return best

    def illa_mcs(self, model, snr_db, candidates=None):
        """Highest-SE MCS whose BLER is at or below ``bler_target`` (ILLA).

        Falls back to the MCS with the lowest BLER if none meet the target.
        """
        cands = self.mcs_indices if candidates is None else list(candidates)
        under = [m for m in cands if self.bler(model, m, snr_db) <= self.bler_target]
        if under:
            return max(under, key=lambda m: self.se[m])
        return min(cands, key=lambda m: self.bler(model, m, snr_db))
