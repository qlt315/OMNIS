"""GDO: Acc-UCB among QoS-feasible arms; radio/GPU via ES BCD. No Lyapunov drift."""

from __future__ import annotations

import time

import numpy as np

from baselines.slot_env import SlotEnv
from omnis.radio_obs import radio_grant_wait
from omnis.task_pipeline import STAGE_TX
from sys_data.config import Config


class GDO(SlotEnv):
    def __init__(self, config):
        super().__init__(config)
        self.name = "gdo"
        self.gdo_price_radio = float(getattr(config, "gdo_price_radio", 1.0))
        self.gdo_price_compute = float(getattr(config, "gdo_price_compute", 1.0))
        self.gdo_offer = float(getattr(config, "gdo_offer", 2.0))
        self.gdo_mab_beta = float(getattr(config, "gdo_mab_beta", 0.25))
        # Scale predicted delay/energy limits: <1 tightens the QoS gate.
        self.gdo_feas_margin = float(getattr(config, "gdo_feas_margin", 1.0))
        self._pr = float(self.gdo_price_radio)
        self._pc = float(self.gdo_price_compute)
        self._slice = {}
        self._last_parallel_decision_s = 0.0
        self._last_parallel_update_s = 0.0
        self._acc_sum = {m["name"]: 0.0 for m in self.models}
        self._acc_n = {m["name"]: 0 for m in self.models}
        self._acc_t = 0

    def learn_after_slot(self, pend):
        """Update per-branch Acc UCB from completion labels (env Acc only)."""
        t0 = time.time()
        acc_dic = pend.get("acc") or {}
        msel = pend.get("model_selection") or {}
        for user in pend.get("users") or list(acc_dic.keys()):
            model = (msel.get(user) or {}).get("model")
            if not model or model not in self._acc_n:
                continue
            self._acc_sum[model] += float(acc_dic[user])
            self._acc_n[model] += 1
            self._acc_t += 1
        self._last_parallel_update_s = time.time() - t0

    def _forecast_uplink_bw(self, user, cell_id, last_bw):
        """Same association-share BW forecast as OMNIS+."""
        del last_bw
        if cell_id is None:
            granted = self._last_bandwidth.get(user)
            if granted is None:
                return float(self.total_bandwidth) / max(self.user_num, 1)
            return float(granted)
        state = getattr(self, "_cell_compute_state", None) or {}
        n_assoc = int(state.get(int(cell_id), {}).get("n_assoc", 0))
        share = float(self.total_bandwidth) / max(n_assoc + 1, 1)
        granted = self._last_bandwidth.get(user)
        if granted is None or float(granted) <= 1.0:
            return share
        return min(float(granted), share)

    def predict_md_overheads(self, user, rate_m, model_name, mcs_idx, snr_db=0.0,
                             cell_id=None):
        """OMNIS+-aligned sojourn: service + Q^j·edge + grant wait."""
        del rate_m
        md = self.md_params[user]
        es = self.es_params
        local_d = (
            self.head_flops[model_name] * 1e-9
            / (md["freq"] * md["cores"] * md["flops_per_cycle"]))
        local_e = md["power_coeff"] * md["freq"] ** 3 * local_d
        bw = self._forecast_uplink_bw(
            user, cell_id,
            self._last_bandwidth.get(
                user, self.total_bandwidth / max(self.user_num, 1)))
        se = max(self.mcs_table.delay_se(model_name, mcs_idx, snr_db), 1e-9)
        bits = self._payload_bits(model_name)
        trans_d = bits / max(float(bw) * se, 1e-12)
        gpu_hat = max(float(self._gpu_hat_for_association(user, cell_id=cell_id)), 1e-12)
        edge_d = self.tail_flops[model_name] * 1e-9 / (
            gpu_hat * es["cores"] * es["flops_per_cycle"])
        edge_e = es["power_coeff"] * gpu_hat ** 3 * edge_d
        service = local_d + trans_d + edge_d
        q_j = float(self.pipeline.jobs_ahead(user, cell_id=cell_id))
        grant = radio_grant_wait(local_d, self.slot_duration)
        sojourn = service + q_j * edge_d + grant
        energy = local_e + md["trans_power"] * (trans_d + grant) + edge_e
        return service, sojourn, energy

    def _mcs_for_snr(self, model_name, snr_db):
        bler_t = getattr(self, "bler_target", self.mcs_table.bler_target)
        best_mcs, best_se = self.available_mcs[0], -1.0
        fallback_mcs, fallback_se = best_mcs, -1.0
        for mcs in self.available_mcs:
            se = float(self.mcs_table.delay_se(model_name, mcs, snr_db))
            if se > fallback_se:
                fallback_mcs, fallback_se = mcs, se
            if self.mcs_table.bler(model_name, mcs, snr_db) <= bler_t and se > best_se:
                best_mcs, best_se = mcs, se
        return best_mcs if best_se >= 0.0 else fallback_mcs

    def _pred_acc(self, model_name):
        """UCB Acc from observations only (never the offline Acc table)."""
        n = int(self._acc_n.get(model_name, 0))
        if n <= 0:
            return 1.0
        mu = float(self._acc_sum[model_name]) / n
        t = max(int(self._acc_t), 1)
        bonus = self.gdo_mab_beta * float(np.sqrt(2.0 * np.log(t + 1.0) / n))
        return float(np.clip(mu + bonus, 0.0, 1.0))

    def _eg(self, bfrac, gfrac, radio_occ):
        profit = self.gdo_offer - self._pr * bfrac - self._pc * gfrac
        if profit <= 0.0:
            return -np.inf
        if radio_occ <= 1e-12:
            denom = bfrac + gfrac
            if denom <= 1e-12:
                return -np.inf
            return profit * np.sqrt(2.0) / denom
        denom = bfrac * radio_occ
        if denom <= 1e-12:
            return -np.inf
        return profit / bfrac

    def _radio_occupation(self):
        radio = np.zeros(self.num_cells, dtype=float)
        cap_bw = max(float(self.total_bandwidth), 1e-12)
        for user, task in self.pipeline.active.items():
            if task is None or task.cell_id is None:
                continue
            if task.stage != STAGE_TX:
                continue
            bw = float(self._slice.get(user, {}).get("bw", 0.0))
            if bw <= 0.0:
                bw = float(self._last_bandwidth.get(user, 0.0))
            radio[int(task.cell_id)] += bw / cap_bw
        return radio

    def _propose(self, user, task, cand_cells, sinr_db, radio_occ):
        """Max Acc-UCB among QoS-feasible arms; least-violation fallback."""
        best = None
        best_miss = None
        n_cells = min(self.top_l_cells, len(cand_cells))
        pool = max(float(self.total_bandwidth), 1e-12)
        gpu_pool = max(float(self.es_params["freq"]), 1e-12)
        d_lim = float(task["delay_constraint"]) * self.gdo_feas_margin
        e_lim = float(task["energy_constraint"]) * self.gdo_feas_margin
        for rank in range(n_cells):
            cell_id = int(cand_cells[rank])
            snr_db_c = float(sinr_db[cell_id])
            residual = 1.0 - float(radio_occ[cell_id])
            if residual <= 1e-9:
                continue
            for model in self.models:
                model_name = model["name"]
                mcs = self._mcs_for_snr(model_name, snr_db_c)
                _svc, soj, energy = self.predict_md_overheads(
                    user, None, model_name, mcs, snr_db=snr_db_c,
                    cell_id=cell_id)
                bw_hat = self._forecast_uplink_bw(
                    user, cell_id,
                    self._last_bandwidth.get(user, pool / max(self.user_num, 1)))
                bfrac = float(bw_hat) / pool
                gfrac = float(np.clip(
                    float(self._gpu_hat_for_association(user, cell_id=cell_id))
                    / gpu_pool, 1e-9, 1.0))
                if bfrac > residual + 1e-12:
                    excess = (max(0.0, soj - d_lim) + max(0.0, energy - e_lim)
                              + 10.0 * (bfrac - residual))
                    acc = self._pred_acc(model_name)
                    miss_key = (-excess, float(acc), -float(soj),
                                -float(self._payload_bits(model_name)))
                    if best_miss is None or miss_key > best_miss[0]:
                        best_miss = (
                            miss_key, rank, cell_id, model_name,
                            min(bfrac, residual), gfrac, False)
                    continue
                if soj <= d_lim and energy <= e_lim:
                    eg = self._eg(bfrac, gfrac, radio_occ[cell_id])
                    if not np.isfinite(eg):
                        eg = -1e18
                    acc = self._pred_acc(model_name)
                    key = (float(acc), float(eg), -float(soj), -float(bfrac))
                    if best is None or key > best[0]:
                        best = (key, rank, cell_id, model_name,
                                bfrac, gfrac, True)
                    continue
                excess = max(0.0, soj - d_lim) + max(0.0, energy - e_lim)
                acc = self._pred_acc(model_name)
                miss_key = (-excess, float(acc), -float(soj),
                            -float(self._payload_bits(model_name)))
                if best_miss is None or miss_key > best_miss[0]:
                    best_miss = (
                        miss_key, rank, cell_id, model_name,
                        bfrac, gfrac, False)
        return best if best is not None else best_miss

    def model_selection(self, context_dic, task_dic, cand_cells_dic,
                        sinr_db_all_dic):
        """Admit highest Acc-UCB QoS-feasible tasks; BCD assigns air resources."""
        del context_dic
        t0 = time.time()
        model_selection_dic = {}
        cell_dic = {}
        remaining = []
        user_index = {user: i for i, user in enumerate(self.pipeline.users)}
        for user in self.users:
            locked_cell = self.pipeline.locked_cell(user)
            locked_model = self.pipeline.locked_model(user)
            if locked_cell is not None and locked_model is not None:
                rank = (cand_cells_dic[user].index(locked_cell)
                        if locked_cell in cand_cells_dic[user] else 0)
                locked_idx = next(
                    (i for i, m in enumerate(self.models)
                     if m["name"] == locked_model), 0)
                model_selection_dic[user] = {
                    "model": locked_model, "cell_rank": rank}
                cell_dic[user] = locked_cell
                ui = user_index.get(user)
                if ui is not None:
                    self.action_freq[
                        ui, locked_idx, min(rank, self.top_l_cells - 1)] += 1
                continue
            remaining.append(user)
            self._slice.pop(user, None)

        radio_occ = self._radio_occupation()
        pool = float(self.total_bandwidth)
        gpu_pool = float(self.es_params["freq"])

        props = {
            user: self._propose(
                user, task_dic[user], cand_cells_dic[user],
                sinr_db_all_dic[user], radio_occ)
            for user in remaining
        }
        pending = set(remaining)
        while pending:
            pool_rows = []
            for user in pending:
                prop = props.get(user)
                if prop is None:
                    continue
                _key, rank, cell_id, model_name, bfrac, gfrac, ok = prop
                if ok and float(bfrac) > 1.0 - float(radio_occ[cell_id]) + 1e-12:
                    prop = self._propose(
                        user, task_dic[user], cand_cells_dic[user],
                        sinr_db_all_dic[user], radio_occ)
                    props[user] = prop
                    if prop is None:
                        continue
                    _key, rank, cell_id, model_name, bfrac, gfrac, ok = prop
                pool_rows.append((user, prop))
            feasible = [(u, p) for u, p in pool_rows if p is not None and p[6]]
            rows = feasible or [(u, p) for u, p in pool_rows if p is not None]
            if not rows:
                for user in list(pending):
                    model_selection_dic[user] = {
                        "model": "", "cell_rank": 0, "defer": True}
                    cell_dic[user] = int(cand_cells_dic[user][0])
                break
            user, prop = max(rows, key=lambda item: item[1][0])
            _key, rank, cell_id, model_name, bfrac, gfrac, ok = prop
            model_idx = next(
                i for i, m in enumerate(self.models) if m["name"] == model_name)
            model_selection_dic[user] = {"model": model_name, "cell_rank": rank}
            cell_dic[user] = cell_id
            self._slice[user] = {
                "bw": float(bfrac) * pool,
                "gpu": float(gfrac) * gpu_pool,
            }
            ui = user_index.get(user)
            if ui is not None:
                self.action_freq[
                    ui, model_idx, min(rank, self.top_l_cells - 1)] += 1
            if ok:
                radio_occ[cell_id] += float(bfrac)
            pending.discard(user)

        self._last_parallel_decision_s = time.time() - t0
        return model_selection_dic, cell_dic


if __name__ == "__main__":
    cfg = Config(0)
    cfg.time_slot_num = 30
    cfg.update_users(6)
    agent = GDO(cfg)
    t0 = time.time()
    agent.simulation()
    print("wall", time.time() - t0)
    print("acc", agent.average_metrics.get("accuracy"),
          "vio", agent.average_metrics.get("vio_prob"))
