"""GDO baseline: online empirical Acc learner + SEM-O-RAN–style Acc-floor greedy.

Name kept as GDO for continuity with prior OMNIS baselines. This is **not** an
oracle Acc-table policy:

  • Start exploratory (random model/cell for ``gdo_explore_slots``).
  • Maintain running-mean Acc (and counts) per model from realized observations.
  • After burn-in + enough samples: Acc-floor on **empirical** Acc → pick the
    lightest model among those meeting ``gdo_acc_floor``; else keep exploring.
  • Cell association: best-cell among Top-L after z* is fixed; EG admission
    uses empirical Acc as the offer (SEM-O-RAN spirit).

Deliberate gaps vs Causal (so GDO stays a weaker literature-style baseline):
  1. No Lyapunov / DPP primary score — myopic Acc-floor + EG only.
  2. Channel-agnostic z* from empirical Acc (not residual GP / SINR features).
  3. Best-cell association only after z* is fixed.
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
        self.gdo_acc_floor = float(getattr(
            config, "gdo_acc_floor", getattr(config, "sem_acc_floor", 0.25)))
        self.gdo_price_radio = getattr(
            config, "gdo_price_radio", getattr(config, "sem_price_radio", 1.0))
        self.gdo_price_compute = getattr(
            config, "gdo_price_compute", getattr(config, "sem_price_compute", 1.0))
        self.gdo_offer_scale = getattr(
            config, "gdo_offer_scale", getattr(config, "sem_offer_scale", 1.0))
        # Initial 乱搞: random model/cell for this many slots.
        self.gdo_explore_slots = int(getattr(config, "gdo_explore_slots", 20))
        # Min observations per model before empirical Acc-floor is trusted.
        self.gdo_min_samples = int(getattr(config, "gdo_min_samples", 5))
        # Optional ε after burn-in (keep some exploration).
        self.gdo_epsilon = float(getattr(config, "gdo_epsilon", 0.05))
        self._max_payload = max(self.data_size.values())
        self._max_tail = max(max(self.tail_flops.values()), 1.0)
        self._last_task = None
        self._last_sinr_db_all = None
        self._last_model_selection = None
        self._slot = 0
        self._emp_acc_sum = {m["name"]: 0.0 for m in self.models}
        self._emp_acc_n = {m["name"]: 0 for m in self.models}
        self.static_model_dic = {}
        self.static_cell_rank_dic = {}

    def get_trans_rate(self, t):
        out = super().get_trans_rate(t)
        # Index 3 = estimated SINR (association / decisions); 4 = true (env).
        self._last_sinr_db_all = out[3]
        return out

    def generate_tasks(self, time_slot):
        self._last_task = super().generate_tasks(time_slot)
        return self._last_task

    def empirical_accuracy(self, model_name):
        n = self._emp_acc_n[model_name]
        if n <= 0:
            return None
        return self._emp_acc_sum[model_name] / n

    def _z_star_model(self):
        """Lightest payload whose **empirical** Acc meets Ac (else None → explore)."""
        models_sorted = sorted(
            self.models, key=lambda m: self.data_size[m["name"]])
        for cand in models_sorted:
            name = cand["name"]
            if self._emp_acc_n[name] < self.gdo_min_samples:
                continue
            emp = self.empirical_accuracy(name)
            if emp is not None and emp + 1e-9 >= self.gdo_acc_floor:
                return name
        return None

    def _resource_vector(self, model_name, snr_db, n_on_cell, mcs_idx=None):
        # PHY BLER/SE only (not Acc table). Mid MCS if unspecified.
        if mcs_idx is None:
            mcs_idx = self.available_mcs[len(self.available_mcs) // 3]
        se = max(self.mcs_table.goodput_se(model_name, mcs_idx, snr_db), 1e-9)
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

    def _random_selection(self, cand_cells_dic):
        model_selection_dic, cell_dic = {}, {}
        n_models = len(self.models)
        for user_idx, user in enumerate(self.users):
            top = cand_cells_dic[user]
            L = min(self.top_l_cells, len(top))
            model_idx = int(np.random.randint(0, n_models))
            cell_rank = int(np.random.randint(0, L))
            model_selection_dic[user] = {
                "model": self.models[model_idx]["name"],
                "cell_rank": cell_rank,
            }
            cell_dic[user] = top[cell_rank]
            self.action_freq[user_idx, model_idx, cell_rank] += 1
        return model_selection_dic, cell_dic

    def _gdo_greedy(self, task_dic, cand_cells_dic, sinr_db_all_dic, z_model):
        """Empirical Acc-floor z* + best-cell + offer/price EG admission."""
        emp = self.empirical_accuracy(z_model)
        z_acc = 0.5 if emp is None else float(emp)
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
        explore = (
            self._slot < self.gdo_explore_slots
            or (self.gdo_epsilon > 0 and np.random.rand() < self.gdo_epsilon)
        )
        z_model = None if explore else self._z_star_model()
        if z_model is None:
            out = self._random_selection(cand_cells_dic)
        else:
            out = self._gdo_greedy(
                self._last_task, cand_cells_dic, self._last_sinr_db_all, z_model)
        self._last_model_selection = out[0]
        self._slot += 1
        return out

    def observe_accuracy(self, model_selection_dic, acc_dic):
        """Update empirical Acc from env-realized observations (not table queries)."""
        for user in self.users:
            name = model_selection_dic[user]["model"]
            self._emp_acc_sum[name] += float(acc_dic[user])
            self._emp_acc_n[name] += 1

    def get_instant_metrics(self, task_dic, total_overhead_dic, reward_dic, acc_dic,
                            queue_info_dic=None):
        out = super().get_instant_metrics(
            task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic)
        if self._last_model_selection is not None:
            self.observe_accuracy(self._last_model_selection, acc_dic)
        return out


if __name__ == "__main__":
    c = Config(0)
    c.update_users(6)
    c.time_slot_num = 40
    agent = GDO(c)
    t0 = time.time()
    agent.simulation()
    print("aver:", agent.average_metrics, "wall", time.time() - t0)
    print("gdo_acc_floor=", agent.gdo_acc_floor,
          "explore_slots=", agent.gdo_explore_slots,
          "emp_n=", agent._emp_acc_n,
          "emp_acc=",
          {m: (None if agent.empirical_accuracy(m) is None
               else round(agent.empirical_accuracy(m), 4))
           for m in agent._emp_acc_n})
