"""Unused myopic V·u+drift oracle (ablation / archive).

GDO in ``gdo_main.py`` is the literature SF-ESP Acc-floor greedy.
This module keeps the previous queue-aware argmax(V·u+drift) scoring for
optional ablation — not imported by sweeps or train_all.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from baselines.rss_main import RSS


class MyopicDPP(RSS):
    """Per-user argmax of V·utility + Lyapunov drift (no Acc-floor primary)."""

    def __init__(self, config):
        super().__init__(config)
        self.name = "oracle"
        self._last_task = None
        self._last_sinr_db_all = None
        self.static_model_dic = {}
        self.static_cell_rank_dic = {}

    def get_trans_rate(self, t):
        out = super().get_trans_rate(t)
        self._last_sinr_db_all = out[3]
        return out

    def generate_tasks(self, time_slot):
        self._last_task = super().generate_tasks(time_slot)
        return self._last_task

    def _arm_score(self, user, model_name, snr_db, task_u):
        mcs_hat = self.forward_sim_mcs(user, snr_db, model_name, task_u)
        _svc, sojourn_hat, energy_hat = self.predict_md_overheads(
            user, None, model_name, mcs_hat, snr_db=snr_db)
        acc_hat = float(self.mcs_table.accuracy(model_name, mcs_hat, snr_db))
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
                    score = self._arm_score(user, mname, snr_db, task_u)
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
        return self._greedy(
            self._last_task, cand_cells_dic, self._last_sinr_db_all)
