"""Unused myopic V·u+drift oracle (ablation / archive).

GDO in ``gdo_main.py`` is the online empirical Acc-floor + SF-ESP greedy.
This module keeps queue-aware argmax(V·u+drift) scoring for optional ablation
— not imported by sweeps or convergence_all. Acc estimates use last realized Acc
per model (or 0.5 before any obs); never the Acc table at decision time.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from baselines.rss_main import RSS


class MyopicDPP(RSS):
    """Per-user argmax of V·utility + Lyapunov drift (no Acc-table oracle)."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "oracle"
        self._last_task = None
        self._last_sinr_db_all = None
        self._last_model_selection = None
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

    def _emp_acc(self, model_name):
        n = self._emp_acc_n[model_name]
        if n <= 0:
            return 0.5  # uninformative until observations arrive
        return self._emp_acc_sum[model_name] / n

    def _arm_score(self, user, model_name, snr_db, task_u, cell_id=None):
        mcs_hat = self.forward_sim_mcs(user, snr_db, model_name, task_u, cell_id=cell_id)
        _svc, sojourn_hat, energy_hat = self.predict_md_overheads(
            user, None, model_name, mcs_hat, snr_db=snr_db, cell_id=cell_id)
        acc_hat = self._emp_acc(model_name)
        w_acc = getattr(self, "reward_w_acc", 1.0)
        qos = getattr(self, "reward_qos_coef", 1.5)
        utility = (
            w_acc * acc_hat
            + qos * task_u["delay_weight"] * erf(
                task_u["delay_constraint"] - sojourn_hat)
            + qos * task_u["energy_weight"] * erf(
                task_u["energy_constraint"] - energy_hat))
        drift = self.dpp_drift(
            user, model_name, mcs_hat, energy_hat, snr_db=snr_db)
        return self.lyapunov_v * utility + drift

    def _greedy(self, task_dic, cand_cells_dic, sinr_db_all_dic):
        model_selection_dic, cell_dic = {}, {}
        name_to_idx = {m["name"]: i for i, m in enumerate(self.models)}
        for user_idx, user in enumerate(self.users):
            top_cells = cand_cells_dic[user]
            sinr_db_all = sinr_db_all_dic[user]
            task_u = task_dic[user]
            best_val = -np.inf
            best_model_name = self.models[0]["name"]
            best_rank = 0
            best_cell = top_cells[0]
            for model in self.models:
                mname = model["name"]
                for cell_rank in range(min(self.top_l_cells, len(top_cells))):
                    cell_id = top_cells[cell_rank]
                    snr_db = float(sinr_db_all[cell_id])
                    score = self._arm_score(user, mname, snr_db, task_u, cell_id=cell_id)
                    if score > best_val:
                        best_val = score
                        best_model_name = mname
                        best_rank = cell_rank
                        best_cell = cell_id
            model_selection_dic[user] = {
                "model": best_model_name, "cell_rank": best_rank}
            cell_dic[user] = best_cell
            self.action_freq[user_idx, name_to_idx[best_model_name], best_rank] += 1
        return model_selection_dic, cell_dic

    def model_selection(self, cand_cells_dic):
        out = self._greedy(
            self._last_task, cand_cells_dic, self._last_sinr_db_all)
        self._last_model_selection = out[0]
        return out

    def get_instant_metrics(self, task_dic, total_overhead_dic, reward_dic, acc_dic,
                            queue_info_dic=None):
        out = super().get_instant_metrics(
            task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic)
        if self._last_model_selection is not None:
            for user in self.users:
                name = self._last_model_selection[user]["model"]
                self._emp_acc_sum[name] += float(acc_dic[user])
                self._emp_acc_n[name] += 1
        return out
