"""GDO baseline: SEM-O-RAN–inspired SF-ESP greedy (Puligheddu et al., TMC 2024).

Name kept as GDO for continuity with prior OMNIS baselines; the algorithm is the
SF-ESP Acc-floor greedy adapted to our (model, cell_rank) space:

  • Compression factor z  ↔  split-DNN payload (``data_size``)
  • Accuracy floor Ac     ↔  ``gdo_acc_floor`` (default 0.25 — Acc-chasing /
    Box12-locking literature baseline)
  • Primary selection     ↔  lightest model with offline a(z) ≥ Ac
                            (``min z s.t. Acc ≥ Ac``)
  • Resource vector s     ↔  (radio load ∝ payload/SE, compute ∝ tail FLOPs)
  • Effective Gradient    ↔  Eq. (2) multidimensional-knapsack heuristic
  • Cell association      ↔  best-cell among top-L after z* is fixed

Deliberate gaps vs Causal (so GDO stays a weaker literature baseline):
  1. **No Lyapunov / DPP** — myopic admission only (not V·u+drift primary score).
  2. **Channel-agnostic z*** — offline accuracy–compression curve at ref SNR/MCS.
  3. **Best-cell association only** after z* is fixed.
  4. **Rigid per-class profile** — no flexible upgrade to heavier models.

(Myopic V·u+drift scoring is intentionally *not* used here; that oracle lives
elsewhere if needed for ablation.)
"""

from __future__ import annotations

import time

import numpy as np

from baselines.rss_main import RSS
from sys_data.config import Config


class GDO(RSS):
    def __init__(self, config):
        super().__init__(config)
        self.name = "gdo"
        # Accept gdo_* (preferred) or legacy sem_* config keys.
        # Default 0.25 restores Acc-floor / Box12-locking SEM-O-RAN baseline.
        self.gdo_acc_floor = float(getattr(
            config, "gdo_acc_floor", getattr(config, "sem_acc_floor", 0.25)))
        self.gdo_ref_snr_db = float(getattr(
            config, "gdo_ref_snr_db", getattr(config, "sem_ref_snr_db", 5.0)))
        self.gdo_ref_mcs = getattr(
            config, "gdo_ref_mcs", getattr(config, "sem_ref_mcs", None))
        self.gdo_price_radio = getattr(
            config, "gdo_price_radio", getattr(config, "sem_price_radio", 1.0))
        self.gdo_price_compute = getattr(
            config, "gdo_price_compute", getattr(config, "sem_price_compute", 1.0))
        self.gdo_offer_scale = getattr(
            config, "gdo_offer_scale", getattr(config, "sem_offer_scale", 1.0))
        self._max_payload = max(self.data_size.values())
        self._max_tail = max(max(self.tail_flops.values()), 1.0)
        self._last_task = None
        self._last_sinr_db_all = None
        self.static_model_dic = {}
        self.static_cell_rank_dic = {}
        if self.gdo_ref_mcs is None:
            self.gdo_ref_mcs = self.available_mcs[len(self.available_mcs) // 3]

    def get_trans_rate(self, t):
        out = super().get_trans_rate(t)
        self._last_sinr_db_all = out[3]
        return out

    def generate_tasks(self, time_slot):
        self._last_task = super().generate_tasks(time_slot)
        return self._last_task

    def _offline_accuracy(self, model_name):
        """Offline a(z) at conservative reference SNR/MCS (Acc-floor curve)."""
        return float(self.mcs_table.accuracy(
            model_name, self.gdo_ref_mcs, self.gdo_ref_snr_db))

    def _z_star_model(self):
        """Lightest payload whose offline accuracy meets Ac (min z s.t. Acc≥Ac)."""
        models_sorted = sorted(
            self.models, key=lambda m: self.data_size[m["name"]])
        for cand in models_sorted:
            if self._offline_accuracy(cand["name"]) + 1e-9 >= self.gdo_acc_floor:
                return cand["name"]
        # None meet floor → heaviest Acc-chasing fallback (literature Acc baseline)
        return models_sorted[-1]["name"]

    def _resource_vector(self, model_name, snr_db, n_on_cell):
        se = max(self.mcs_table.goodput_se(
            model_name, self.gdo_ref_mcs, snr_db), 1e-9)
        radio = (
            (self.data_size[model_name] / self._max_payload)
            * (1.0 + 0.25 * n_on_cell) / se)
        compute = self.tail_flops[model_name] / self._max_tail
        return np.array([radio, max(compute, 0.05)], dtype=float)

    def _effective_gradient(self, offer, cost, s, occupied, capacity):
        """SEM-O-RAN Eq. (2)–style multidimensional knapsack EG heuristic."""
        profit = offer - cost
        if profit <= 0:
            return -np.inf
        if np.all(occupied <= 1e-12):
            denom = float(np.sum(s / capacity))
            return -np.inf if denom <= 1e-12 else profit * np.sqrt(len(s)) / denom
        denom = float(np.sum(s * occupied / capacity))
        return -np.inf if denom <= 1e-12 else (
            profit * np.sqrt(float(np.sum(occupied ** 2))) / denom)

    def _gdo_greedy(self, task_dic, cand_cells_dic, sinr_db_all_dic):
        """Acc-floor z* + best-cell + offer/price EG admission order."""
        z_model = self._z_star_model()
        z_acc = self._offline_accuracy(z_model)
        offer = self.gdo_offer_scale * z_acc
        prices = np.array(
            [self.gdo_price_radio, self.gdo_price_compute], dtype=float)
        capacity = np.array(
            [float(self.user_num), float(self.user_num)], dtype=float)
        occupied = np.zeros(2, dtype=float)
        cell_load = {}

        candidates = {}
        for user in self.users:
            best_rank, best_cell, best_snr = 0, cand_cells_dic[user][0], -1e9
            for rank in range(min(self.top_l_cells, len(cand_cells_dic[user]))):
                cell_id = cand_cells_dic[user][rank]
                snr_db = float(sinr_db_all_dic[user][cell_id])
                if snr_db > best_snr:
                    best_snr, best_rank, best_cell = snr_db, rank, cell_id
            candidates[user] = {
                "model": z_model, "cell_rank": best_rank, "cell_id": best_cell,
                "snr_db": best_snr, "offer": offer,
            }

        remaining = set(self.users)
        model_selection_dic, cell_dic = {}, {}
        model_idx = next(
            i for i, m in enumerate(self.models) if m["name"] == z_model)

        while remaining:
            best_user, best_meta, best_eg = None, None, -np.inf
            for user in remaining:
                meta = candidates[user]
                s = self._resource_vector(
                    meta["model"], meta["snr_db"],
                    cell_load.get(meta["cell_id"], 0))
                cost = float(np.dot(prices, s))
                eg = self._effective_gradient(
                    meta["offer"], cost, s, occupied, capacity)
                if eg > best_eg:
                    best_eg = eg
                    best_user = user
                    best_meta = {**meta, "s": s}
            if best_user is None:
                best_user = remaining.pop()
                meta = candidates[best_user]
                model_selection_dic[best_user] = {
                    "model": meta["model"], "cell_rank": meta["cell_rank"]}
                cell_dic[best_user] = meta["cell_id"]
                ui = self.users.index(best_user)
                self.action_freq[ui, model_idx, meta["cell_rank"]] += 1
                continue

            model_selection_dic[best_user] = {
                "model": best_meta["model"],
                "cell_rank": best_meta["cell_rank"]}
            cell_dic[best_user] = best_meta["cell_id"]
            occupied = np.minimum(capacity, occupied + best_meta["s"])
            cell_load[best_meta["cell_id"]] = (
                cell_load.get(best_meta["cell_id"], 0) + 1)
            remaining.remove(best_user)
            ui = self.users.index(best_user)
            self.action_freq[ui, model_idx, best_meta["cell_rank"]] += 1

        return model_selection_dic, cell_dic

    def model_selection(self, cand_cells_dic):
        return self._gdo_greedy(
            self._last_task, cand_cells_dic, self._last_sinr_db_all)


if __name__ == "__main__":
    c = Config(0)
    c.update_users(6)
    c.time_slot_num = 40
    agent = GDO(c)
    t0 = time.time()
    agent.simulation()
    print("aver:", agent.average_metrics, "wall", time.time() - t0)
    # Quick Acc-floor sanity: which z* under default floor?
    print("gdo_acc_floor=", agent.gdo_acc_floor,
          "z*=", agent._z_star_model(),
          "offline a=",
          {m["name"]: round(agent._offline_accuracy(m["name"]), 4)
           for m in agent.models})
