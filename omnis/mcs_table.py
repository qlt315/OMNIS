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
        self._se_arr = np.array(
            [self.se[m] for m in self.mcs_indices], dtype=float)

        self.acc_clean = {}
        with open(os.path.join(table_dir, "acc_clean.csv")) as f:
            for r in csv.DictReader(f):
                self.acc_clean[r["model"]] = float(r["accuracy"])

        self._bler = self._load_curves("bler_table.csv", "bler")
        # Decision-time caches: identical (model, mcs, snr) lookups dominate
        # U×M×L×|MCS| arm scoring; table is immutable so cache is always valid.
        self._bler_cache = {}
        self._goodput_cache = {}
        self._all_mcs_cache = {}  # (model, snr_q) -> (bler[M], goodput[M])
        self._build_dense_grids()

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

    def _build_dense_grids(self):
        """Pre-sample BLER on the native 1 dB grid for O(1) nearest-bin lookup."""
        self._model_ids = {m: i for i, m in enumerate(sorted(self._bler.keys()))}
        # Infer SNR grid from first curve.
        any_model = next(iter(self._bler))
        any_mcs = next(iter(self._bler[any_model]))
        snr_grid = self._bler[any_model][any_mcs][0]
        self._snr_min = float(snr_grid[0])
        self._snr_max = float(snr_grid[-1])
        self._snr_step = float(snr_grid[1] - snr_grid[0]) if len(snr_grid) > 1 else 1.0
        n_snr = len(snr_grid)
        n_model = len(self._model_ids)
        n_mcs = len(self.mcs_indices)
        self._bler_grid = np.zeros((n_model, n_mcs, n_snr), dtype=float)
        for model, mi in self._model_ids.items():
            for j, mcs in enumerate(self.mcs_indices):
                snr, val = self._bler[model][mcs]
                self._bler_grid[mi, j, :] = np.clip(
                    np.interp(snr_grid, snr, val), 0.0, 1.0)
        self._goodput_grid = self._bler_grid.copy()
        for j, mcs in enumerate(self.mcs_indices):
            self._goodput_grid[:, j, :] = self.se[mcs] * (1.0 - self._bler_grid[:, j, :])

    def _snr_lerp_weights(self, snr_db):
        """Return (i0, i1, w) for linear interp on the dense SNR grid (clamped)."""
        n = self._bler_grid.shape[2]
        x = (float(snr_db) - self._snr_min) / self._snr_step
        if x <= 0.0:
            return 0, 0, 0.0
        if x >= n - 1:
            return n - 1, n - 1, 0.0
        i0 = int(np.floor(x))
        i1 = i0 + 1
        return i0, i1, float(x - i0)

    def _grid_lookup(self, grid, model, mcs_idx, snr_db):
        mi = self._model_ids[model]
        mcs_i = int(mcs_idx)
        i0, i1, w = self._snr_lerp_weights(snr_db)
        v0 = grid[mi, mcs_i, i0]
        if w == 0.0 or i0 == i1:
            return float(v0)
        return float((1.0 - w) * v0 + w * grid[mi, mcs_i, i1])

    def _interp(self, curves, model, mcs_idx, snr_db):
        snr, val = curves[model][mcs_idx]
        return float(np.interp(snr_db, snr, val))  # clamps outside the grid

    def bler(self, model, mcs_idx, snr_db):
        """Interpolated transport-block error rate."""
        key = (model, int(mcs_idx), float(snr_db))
        hit = self._bler_cache.get(key)
        if hit is not None:
            return hit
        mcs_i = int(mcs_idx)
        if (model in self._model_ids and 0 <= mcs_i < len(self.mcs_indices)
                and self.mcs_indices[mcs_i] == mcs_i):
            val = self._grid_lookup(self._bler_grid, model, mcs_i, snr_db)
        else:
            val = float(np.clip(
                self._interp(self._bler, model, mcs_idx, snr_db), 0.0, 1.0))
        self._bler_cache[key] = val
        return val

    def bler_goodput_all_mcs(self, model, snr_db):
        """Vector BLER and goodput-SE for every MCS at ``snr_db`` (decision hot path)."""
        snr_q = round(float(snr_db), 3)
        key = (model, snr_q)
        hit = self._all_mcs_cache.get(key)
        if hit is not None:
            return hit
        mi = self._model_ids[model]
        i0, i1, w = self._snr_lerp_weights(snr_db)
        if w == 0.0 or i0 == i1:
            bler = self._bler_grid[mi, :, i0].copy()
            goodput = self._goodput_grid[mi, :, i0].copy()
        else:
            bler = (1.0 - w) * self._bler_grid[mi, :, i0] + w * self._bler_grid[mi, :, i1]
            goodput = ((1.0 - w) * self._goodput_grid[mi, :, i0]
                       + w * self._goodput_grid[mi, :, i1])
        self._all_mcs_cache[key] = (bler, goodput)
        return bler, goodput

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
        key = (model, int(mcs_idx), float(snr_db))
        hit = self._goodput_cache.get(key)
        if hit is not None:
            return hit
        mcs_i = int(mcs_idx)
        if (model in self._model_ids and 0 <= mcs_i < len(self.mcs_indices)
                and self.mcs_indices[mcs_i] == mcs_i):
            val = self._grid_lookup(self._goodput_grid, model, mcs_i, snr_db)
        else:
            val = self.se[mcs_idx] * (1.0 - self.bler(model, mcs_idx, snr_db))
        self._goodput_cache[key] = val
        return val

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
