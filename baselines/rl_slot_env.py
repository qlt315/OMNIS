"""Shared slot loop for DQN / PPO / MAPPO."""
from __future__ import annotations

import time

import numpy as np

from baselines.slot_env import SlotEnv
from baselines.rl_nets import RunningMeanStd
from omnis.assoc_info import update_cell_compute_state
from omnis.bcd_loop import run_bcd_slot
from omnis.sim_loop import run_discrete_simulation


class OnlineRLBaseline(SlotEnv):
    """Slot loop with pluggable joint decision + online learning hooks."""

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
        self._last_parallel_decision_s = 0.0
        self._last_parallel_update_s = 0.0
        self.learn_delay_slots = int(getattr(config, "learn_delay_slots", 1) or 0)
        self._pending_learn = None
        self._rl_slot_cache = None

    def local_obs(self, user, task_u, cand_cells, sinr_db_all):
        if user not in cand_cells or user not in sinr_db_all:
            return np.zeros(self.local_obs_dim, dtype=np.float32)
        cands = cand_cells[user]
        sinrs = []
        for r in range(self.top_l):
            if r < len(cands):
                sinrs.append(sinr_db_all[user][cands[r]] / 20.0)
            else:
                sinrs.append(0.0)
        scale = float(getattr(self, "dpp_task_scale", 3.0))
        q_n = self.pipeline.composite_backlog(user) / scale
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
            locked_cell = self.pipeline.locked_cell(user)
            locked_model = self.pipeline.locked_model(user)
            if locked_cell is not None and locked_model is not None:
                model_idx = next(
                    (i for i, m in enumerate(self.models) if m["name"] == locked_model), 0)
                cell_rank = (cand_cells_dic[user].index(locked_cell)
                             if locked_cell in cand_cells_dic[user] else 0)
                self.action_freq[user_idx, model_idx, min(cell_rank, self.top_l - 1)] += 1
                model_selection_dic[user] = {
                    "model": locked_model, "cell_rank": cell_rank}
                cell_dic[user] = locked_cell
                continue
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

    def _flush_pending_learn(self):
        pend = self._pending_learn
        if pend is None:
            self._last_parallel_update_s = 0.0
            return
        t_update = time.time()
        self.learn_after_slot(pend)
        self._last_parallel_update_s = time.time() - t_update
        self._pending_learn = None

    def get_instant_metrics(self, task_dic, total_overhead_dic, reward_dic, acc_dic,
                            queue_info_dic=None):
        out = super().get_instant_metrics(
            task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic)
        # RL learning uses mean reward; cache for post-slot hook.
        slot_mean_reward = float(np.mean(list(reward_dic.values())))
        self.train_rewards.append(slot_mean_reward)
        cache = getattr(self, "_rl_slot_cache", None)
        if cache is not None and not self.eval_mode:
            dpp_targets = {}
            for user in self.users:
                if total_overhead_dic[user]["delay"] <= 0 and acc_dic[user] <= 0:
                    dpp_targets[user] = 0.0
                    continue
                model_name = cache["model_selection"].get(user, {}).get("model")
                if model_name is None:
                    dpp_targets[user] = float(reward_dic[user])
                    continue
                snr_db_u = 10 * np.log10(max(cache["snr_true"].get(user, 1e-12), 1e-12))
                mcs = cache["phy"].get(user, self.available_mcs[0])
                cell_id = cache["cell"].get(user)
                _, _, energy_hat = self.predict_md_overheads(
                    user, None, model_name, mcs, snr_db=snr_db_u, cell_id=cell_id)
                drift = self.dpp_drift(
                    user, model_name, mcs, energy_hat, snr_db=snr_db_u)
                dpp_targets[user] = self.lyapunov_v * float(reward_dic[user]) + drift
            team_r_raw = float(np.mean(list(dpp_targets.values()))) if self.dpp_reward \
                else slot_mean_reward
            team_r = self.normalize_team_r(team_r_raw)
            t = cache["t"]
            slot_info = {
                "t": t,
                "done": 1.0 if t == self.time_slot_num - 1 else 0.0,
                "dpp_targets": dpp_targets,
                "reward_dic": reward_dic,
                "team_r": team_r,
                "team_r_raw": team_r_raw,
                "next_global": self.global_state(
                    task_dic, cache["cand"], cache["sinr_est"]),
                "next_local": {
                    u: self.local_obs(u, task_dic[u], cache["cand"], cache["sinr_est"])
                    for u in self.users
                },
                "task_dic": task_dic,
                "cand_cells_dic": cache["cand"],
                "sinr_db_all_dic": cache["sinr_est"],
            }
            if self.learn_delay_slots > 0:
                self._pending_learn = slot_info
            else:
                t0 = time.time()
                self.learn_after_slot(slot_info)
                self._last_parallel_update_s = time.time() - t0
                self.update_time += self._last_parallel_update_s
        return out

    def select_actions_wrapped(self, cand_cells_dic, task_dic, sinr_db_all_dic, t):
        """select_actions + stash CSI for RL learning hook."""
        # Learning flush is owned by sim_loop (avoid double update timing).
        t_decision = time.time()
        model_selection_dic, cell_dic = self._select_actions_impl(
            cand_cells_dic, task_dic, sinr_db_all_dic, t)
        # sim_loop folds _last_parallel_decision_s into decision_time.
        self._last_parallel_decision_s = time.time() - t_decision
        self._rl_slot_cache = {
            "t": t,
            "cand": cand_cells_dic,
            "sinr_est": sinr_db_all_dic,
            "model_selection": model_selection_dic,
            "cell": cell_dic,
            "snr_true": {},
            "phy": {},
        }
        return model_selection_dic, cell_dic

    def simulation(self):
        """Reuse discrete task pipeline; route decisions through select_actions."""
        self._select_actions_impl = type(self).select_actions.__get__(self, type(self))
        self.select_actions = self.select_actions_wrapped  # type: ignore
        orig_get_instant = type(self).get_instant_metrics.__get__(self, type(self))

        def _get_instant(task_dic, total_overhead_dic, reward_dic, acc_dic,
                         queue_info_dic=None):
            if self._rl_slot_cache is not None:
                self._rl_slot_cache["phy"] = dict(getattr(self, "_last_mcs", {}) or {})
                snr_true = {}
                for u in self.users:
                    cell = self._rl_slot_cache["cell"].get(u)
                    if cell is None:
                        continue
                    snr_true[u] = 10 ** (
                        self._rl_slot_cache["sinr_est"][u][cell] / 10)
                self._rl_slot_cache["snr_true"] = snr_true
            return orig_get_instant(
                task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic)

        self.get_instant_metrics = _get_instant  # type: ignore
        try:
            run_discrete_simulation(self)
        finally:
            self.select_actions = self._select_actions_impl  # type: ignore
            self.get_instant_metrics = orig_get_instant  # type: ignore

        if (not self.eval_mode and self.learn_delay_slots > 0
                and self._pending_learn is not None):
            self._flush_pending_learn()
