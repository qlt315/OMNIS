"""Shared multi-cell slot loop for online RL baselines (DQN / PPO / MAPPO).

Hooks:
  select_actions(...)  -> model_selection_dic, cell_dic
  learn_after_slot(...)  -> buffer / gradient updates (no-op in eval)
"""

from __future__ import annotations

import time

import numpy as np

from baselines.rss_main import RSS
from baselines.rl_nets import RunningMeanStd
from omnis.bcd_loop import run_bcd_slot


class OnlineRLBaseline(RSS):
    """RSS env loop with pluggable joint decision + online learning hooks."""

    def __init__(self, config):
        super().__init__(config)
        self.top_l = self.top_l_cells
        self.n_models = len(self.models)
        self.n_actions_local = self.n_models * self.top_l
        self.local_obs_dim = 4 + self.top_l + 2  # per-user features
        self.global_state_dim = self.user_num * self.local_obs_dim
        self.dpp_reward = getattr(config, "rl_dpp_reward", True)
        self.eval_mode = getattr(config, "rl_eval", False)
        self.reward_norm = getattr(config, "rl_reward_norm", True)
        self.log_every = getattr(config, "rl_log_every", 50)
        self._rew_rms = RunningMeanStd()
        self.train_rewards = []
        self.decision_time = 0.0
        self.bcd_time = 0.0
        self.bcd_iters = 0.0
        self.update_time = 0.0
        self.static_model_dic = {}
        self.static_cell_rank_dic = {}

    def local_obs(self, user, task_u, cand_cells, sinr_db_all):
        sinrs = [sinr_db_all[user][cand_cells[user][r]] / 20.0
                 for r in range(self.top_l)]
        q_n = self.backlog[user] / self.dpp_bit_scale
        z_n = self.energy_queue[user] / self.dpp_energy_scale
        return np.array([
            task_u["delay_constraint"] / 3.0,
            task_u["energy_constraint"] / 2.0,
            task_u["delay_weight"],
            task_u["energy_weight"],
            *sinrs,
            min(q_n, 5.0) / 5.0,
            min(z_n, 5.0) / 5.0,
        ], dtype=np.float32)

    def global_state(self, task_dic, cand_cells_dic, sinr_db_all_dic):
        parts = [self.local_obs(u, task_dic[u], cand_cells_dic, sinr_db_all_dic)
                 for u in self.users]
        return np.concatenate(parts, axis=0)

    def decode_local_action(self, action):
        model_idx = int(action) // self.top_l
        cell_rank = int(action) % self.top_l
        return model_idx, cell_rank

    def pack_selections(self, actions_by_user, cand_cells_dic):
        model_selection_dic = {}
        cell_dic = {}
        for user_idx, user in enumerate(self.users):
            a = int(actions_by_user[user])
            model_idx, cell_rank = self.decode_local_action(a)
            self.action_freq[user_idx, model_idx, cell_rank] += 1
            model_selection_dic[user] = {
                "model": self.models[model_idx]["name"],
                "cell_rank": cell_rank,
            }
            cell_dic[user] = cand_cells_dic[user][cell_rank]
        return model_selection_dic, cell_dic

    def normalize_team_r(self, team_r: float) -> float:
        if not self.reward_norm or self.eval_mode:
            return team_r
        self._rew_rms.update(team_r)
        return self._rew_rms.normalize(team_r)

    def select_actions(self, cand_cells_dic, task_dic, sinr_db_all_dic, t):
        raise NotImplementedError

    def learn_after_slot(self, slot_info):
        raise NotImplementedError

    def simulation(self):
        self._run_simulation()

    def _run_simulation(self):
        for t in range(self.time_slot_num):
            snr_dic, trans_rate_dic, cand_cells_dic, sinr_db_all_dic = self.get_trans_rate(t)
            task_dic = self.generate_tasks(t)

            # MAPPO: batched actor forward ≈ parallel MD cost (not U× sequential).
            # DQN/PPO: joint/centralized wall (cannot factor across users).
            t_decision = time.time()
            model_selection_dic, cell_dic = self.select_actions(
                cand_cells_dic, task_dic, sinr_db_all_dic, t)
            self.decision_time += time.time() - t_decision
            snr_dic = self._apply_cell_association(cell_dic, sinr_db_all_dic)

            arrival_bits_dic = {
                user: task_dic[user]["n_arrivals"]
                * self.data_size[model_selection_dic[user]["model"]]
                for user in self.users
            }
            local_overhead_dic = self.get_local_overhead(model_selection_dic)

            bcd_out = run_bcd_slot(
                self, task_dic, model_selection_dic, trans_rate_dic,
                local_overhead_dic, snr_dic, cell_dic)
            bandwidth_allocation_dic = bcd_out["bandwidth"]
            gpu_allocation_dic = bcd_out["gpu"]
            phy_choice_dic = bcd_out["phy_choice"]
            total_overhead_dic = bcd_out["total_overhead"]

            for user in self.users:
                self.instant_metrics[user]["mcs"].append(phy_choice_dic[user])
                self.instant_metrics[user]["cell"].append(cell_dic[user])
                snr_db_u = 10 * np.log10(max(snr_dic[user], 1e-12))
                self.instant_metrics[user]["bler"].append(self.mcs_table.bler(
                    model_selection_dic[user]["model"], phy_choice_dic[user], snr_db_u))

            acc_dic = self.get_accuracy(snr_dic, phy_choice_dic, model_selection_dic)
            if self.acc_noise_std > 0:
                acc_dic = {
                    user: float(np.clip(acc + np.random.normal(0.0, self.acc_noise_std), 0.0, 1.0))
                    for user, acc in acc_dic.items()
                }
            reward_dic = self.get_reward(task_dic, acc_dic, total_overhead_dic)

            dpp_targets = {}
            for user in self.users:
                snr_db_u = 10 * np.log10(max(snr_dic[user], 1e-12))
                _, _, energy_hat = self.predict_md_overheads(
                    user, None, model_selection_dic[user]["model"],
                    phy_choice_dic[user], snr_db=snr_db_u)
                drift = self.dpp_drift(
                    user, model_selection_dic[user]["model"],
                    phy_choice_dic[user], energy_hat, snr_db=snr_db_u)
                dpp_targets[user] = self.lyapunov_v * float(reward_dic[user]) + drift

            service_bits_dic = {
                user: bandwidth_allocation_dic[user]
                * self._goodput_se(
                    user, model_selection_dic[user]["model"],
                    phy_choice_dic[user], snr_dic)
                * self.slot_duration
                for user in self.users
            }
            queue_info_dic = {}
            for user in self.users:
                self.backlog[user] = max(self.backlog[user] - service_bits_dic[user], 0.0) \
                    + arrival_bits_dic[user]
                self.energy_queue[user] = max(
                    self.energy_queue[user] + total_overhead_dic[user]["energy"]
                    - self.energy_budget[user], 0.0)
                queue_info_dic[user] = {
                    "backlog": self.backlog[user], "energy_queue": self.energy_queue[user],
                    "arrivals": arrival_bits_dic[user], "served": service_bits_dic[user],
                }
            self.get_instant_metrics(task_dic, total_overhead_dic, reward_dic, acc_dic,
                                     queue_info_dic)
            slot_mean_reward = float(np.mean(list(reward_dic.values())))
            self.train_rewards.append(slot_mean_reward)
            self._last_bandwidth = bandwidth_allocation_dic
            self._last_gpu = gpu_allocation_dic

            if not self.eval_mode:
                t_update = time.time()
                next_state = self.global_state(task_dic, cand_cells_dic, sinr_db_all_dic)
                next_local = {
                    u: self.local_obs(u, task_dic[u], cand_cells_dic, sinr_db_all_dic)
                    for u in self.users
                }
                team_r_raw = float(np.mean(list(dpp_targets.values()))) if self.dpp_reward \
                    else float(np.mean(list(reward_dic.values())))
                team_r = self.normalize_team_r(team_r_raw)
                slot_info = {
                    "t": t,
                    "done": 1.0 if t == self.time_slot_num - 1 else 0.0,
                    "dpp_targets": dpp_targets,
                    "reward_dic": reward_dic,
                    "team_r": team_r,
                    "team_r_raw": team_r_raw,
                    "next_global": next_state,
                    "next_local": next_local,
                    "task_dic": task_dic,
                    "cand_cells_dic": cand_cells_dic,
                    "sinr_db_all_dic": sinr_db_all_dic,
                }
                self.learn_after_slot(slot_info)
                self.update_time += time.time() - t_update

        self.get_average_and_std_metrics()
