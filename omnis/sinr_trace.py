"""Loader for multi-cell SINR traces produced by phy_sim/run_traces.py.

Trace schema: slot, cell_id, ue_id, sinr_db  (all candidate links).
Provides Top-L cell candidates per UE/slot for the joint (model, cell) arm.
"""

import csv
import os

import numpy as np


def select_spread_ue_ids(n_users, num_ues, num_cells):
    """Pick ``n_users`` unique UE indices, round-robin across sites/cells.

    Trace UEs are laid out as ``num_cells`` sites × ``ues_per_site`` UEs
    (e.g. smoke7_sites: 7×6 = 42). Round-robin avoids the old stride
    ``i * ues_per_site`` collision that mapped many users onto the last UE
    when ``n_users > num_cells``, which left SinrTrace locals stuck at −120 dB.
    """
    n_users = int(n_users)
    num_ues = int(num_ues)
    num_cells = max(1, int(num_cells))
    if n_users < 1:
        raise ValueError(f"n_users must be >= 1, got {n_users}")
    if n_users > num_ues:
        raise ValueError(
            f"requested {n_users} users but trace only has {num_ues} unique UEs")
    ues_per_site = max(1, num_ues // num_cells)
    selected = []
    used = set()
    for i in range(n_users):
        site = i % num_cells
        within = i // num_cells
        ue = site * ues_per_site + within
        if ue >= num_ues or ue in used:
            for cand in range(num_ues):
                if cand not in used:
                    ue = cand
                    break
            else:
                raise ValueError("exhausted unique UE pool while spreading users")
        selected.append(int(ue))
        used.add(ue)
    return selected


class SinrTrace:
    def __init__(self, trace_path, link_info_path=None, ue_ids=None):
        """Load a SINR cube.

        Parameters
        ----------
        trace_path : str
            CSV with columns slot, cell_id, ue_id, sinr_db.
        link_info_path : str, optional
            CSV with cell_id, ue_id, distance_m (kept for diagnostics).
        ue_ids : sequence of int, optional
            Subset of trace UE indices to keep (maps onto system users in order).
            Must be unique. If None, keep every UE present in the file.
        """
        rows = []
        with open(trace_path) as f:
            for r in csv.DictReader(f):
                rows.append((int(r["slot"]), int(r["cell_id"]),
                             int(r["ue_id"]), float(r["sinr_db"])))
        if not rows:
            raise ValueError(f"empty SINR trace: {trace_path}")

        all_slots = sorted({r[0] for r in rows})
        all_cells = sorted({r[1] for r in rows})
        all_ues = sorted({r[2] for r in rows})
        if ue_ids is None:
            ue_ids = all_ues
        else:
            ue_ids = list(ue_ids)
            if len(ue_ids) != len(set(ue_ids)):
                raise ValueError(
                    f"ue_ids must be unique (duplicates cause −120 dB fill): {ue_ids}")
            missing = set(ue_ids) - set(all_ues)
            if missing:
                raise ValueError(f"ue_ids not in trace: {missing}")

        self.num_slots = len(all_slots)
        self.num_cells = len(all_cells)
        self.num_ues = len(ue_ids)
        self.cell_ids = list(all_cells)
        self.ue_ids = list(ue_ids)
        self._ue_pos = {u: i for i, u in enumerate(self.ue_ids)}
        self._cell_pos = {c: i for i, c in enumerate(self.cell_ids)}

        # sinr[slot, cell, ue_local]
        self.sinr_db = np.full((self.num_slots, self.num_cells, self.num_ues),
                               -120.0, dtype=np.float64)
        slot_pos = {s: i for i, s in enumerate(all_slots)}
        for slot, cell, ue, snr in rows:
            if ue not in self._ue_pos:
                continue
            self.sinr_db[slot_pos[slot], self._cell_pos[cell], self._ue_pos[ue]] = snr

        self.distances = None
        if link_info_path and os.path.exists(link_info_path):
            dist = np.full((self.num_cells, self.num_ues), np.nan)
            with open(link_info_path) as f:
                for r in csv.DictReader(f):
                    c, u = int(r["cell_id"]), int(r["ue_id"])
                    if c in self._cell_pos and u in self._ue_pos:
                        dist[self._cell_pos[c], self._ue_pos[u]] = float(r["distance_m"])
            self.distances = dist

    def slot_index(self, t):
        """Wrap long simulations over the finite trace."""
        return int(t) % self.num_slots

    def sinr_vector(self, t, ue_local):
        """SINR [dB] from every cell to one UE at slot t."""
        return self.sinr_db[self.slot_index(t), :, ue_local].copy()

    def sinr(self, t, cell, ue_local):
        return float(self.sinr_db[self.slot_index(t), cell, ue_local])

    def top_cells(self, t, ue_local, L):
        """Cell indices of the L strongest links for this UE at slot t."""
        L = min(L, self.num_cells)
        vec = self.sinr_db[self.slot_index(t), :, ue_local]
        return list(np.argsort(vec)[-L:][::-1])

    @staticmethod
    def from_config_dir(table_dir, tag="smoke7", ue_ids=None):
        # Prefer the site-aggregated trace (3 sectors folded per hex site -> one
        # logical cell), which matches the system model's per-cell ES / bandwidth
        # pool abstraction; fall back to the raw sector-level trace if absent.
        site_trace = os.path.join(table_dir, f"sinr_trace_{tag}_sites.csv")
        site_info = os.path.join(table_dir, f"link_info_{tag}_sites.csv")
        if os.path.exists(site_trace):
            return SinrTrace(site_trace,
                             site_info if os.path.exists(site_info) else None,
                             ue_ids=ue_ids)
        trace = os.path.join(table_dir, f"sinr_trace_{tag}.csv")
        info = os.path.join(table_dir, f"link_info_{tag}.csv")
        return SinrTrace(trace, info if os.path.exists(info) else None, ue_ids=ue_ids)
