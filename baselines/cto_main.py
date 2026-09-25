"""Centralized OMNIS+: same Acc GP / score; joint cell admits rescored on a shared BW split."""

import time

from omnis.omnis_main import OMNIS
from omnis.task_pipeline import STAGE_TX
from sys_data.config import Config


class CTO(OMNIS):
    def __init__(self, config):
        # Same mechanism as OMNIS+. Build the accuracy GP even if the caller
        # had selected a reward-GP ablation, then restore their config object.
        prev_algo = getattr(config, "algo", "causal")
        config.algo = "causal"
        super().__init__(config)
        config.algo = prev_algo
        self.algo = "causal"
        self.name = "cto"
        self.centralized = True
        self._bw_override = None
        self._cto_share = {}
        if getattr(self, "causal_mab", None) is not None:
            self.causal_mab.centralized = True

    def _cto_share_bw(self, user, cell_id):
        """Collision split, never above a grant this MD has already observed.

        With no grant yet, the cap is the equal share. pool/n_users is only
        the cold-start placeholder and is not an observed grant.
        """
        share = self._cto_share.get(int(cell_id))
        if share is None:
            return self._forecast_uplink_bw(
                user, cell_id,
                self._last_bandwidth.get(
                    user, self.total_bandwidth / max(self.user_num, 1)))
        granted = self._last_bandwidth.get(user)
        if granted is None or float(granted) <= 1.0:
            return float(share)
        return min(float(granted), float(share))

    def model_selection_causal(self, task_dic, cand_cells_dic, sinr_db_all_dic,
                               trans_rate_dic):
        """Same causal score, then one same-cell rescore under the collision split."""
        del trans_rate_dic
        self._begin_slot_decision_cache()
        es = self.es_params
        model_selection_dic = {}
        cell_dic = {}
        self._bw_override = None
        self._cto_share = {}

        def make_request(user):
            def predict_overheads(model_name, mcs_idx, snr_db=0.0, user=user,
                                  cell_id=None):
                return self.predict_md_overheads(
                    user, None, model_name, mcs_idx, snr_db=snr_db,
                    cell_id=cell_id)

            def drift_score(model_name, mcs_idx, energy_hat, snr_db=0.0,
                            user=user, cell_id=None):
                return self.dpp_drift(
                    user, model_name, mcs_idx, energy_hat,
                    snr_db=snr_db, cell_id=cell_id)

            def parts_for_cell(model_name, cell_id, user=user):
                local_d, local_e, payload, bw, backlog_b, p_tx = (
                    self._oh_user_base[user][model_name])
                if int(cell_id) in self._cto_share:
                    bw = self._cto_share_bw(user, cell_id)
                else:
                    bw = self._forecast_uplink_bw(user, cell_id, bw)
                gpu_hat = self._gpu_hat_for_association(user, cell_id=cell_id)
                edge_d = (self.tail_flops[model_name] * 1e-9
                          / (gpu_hat * es["cores"] * es["flops_per_cycle"]))
                edge_e = es["power_coeff"] * gpu_hat ** 3 * edge_d
                return (local_d, local_e, edge_d, edge_e, payload, bw,
                        backlog_b, p_tx)

            task_u = dict(task_dic[user])
            task_u["backlog_bits"] = float(self.pipeline.composite_backlog(user))
            task_u["dpp_bit_scale"] = float(self.dpp_task_scale)
            task_u["backlog_tasks"] = task_u["backlog_bits"]
            task_u["dpp_task_scale"] = float(self.dpp_task_scale)
            task_u["slot_duration"] = float(self.slot_duration)
            return (
                user, cand_cells_dic[user], sinr_db_all_dic[user],
                task_u, predict_overheads, drift_score, parts_for_cell)

        mab = self.causal_mab
        pending = []
        requests = []
        for user in self.users:
            locked_cell = self.pipeline.locked_cell(user)
            locked_model = self.pipeline.locked_model(user)
            if locked_cell is not None and locked_model is not None:
                model_selection_dic[user] = {
                    "model": locked_model, "cell_rank": 0}
                cell_dic[user] = locked_cell
                continue
            pending.append(user)
            requests.append(make_request(user))

        assigned = {}
        decision_s = 0.0
        try:
            if requests:
                picked = mab.select_arms_batch(requests, advance_slot=True)
                decision_s += float(mab.last_parallel_decision_s)
                for user in pending:
                    model_idx, cell_id = picked[user]
                    assigned[user] = (int(model_idx), int(cell_id))

            by_cell = {}
            for user, (_model_idx, cell_id) in assigned.items():
                by_cell.setdefault(int(cell_id), []).append(user)
            n_tx = {}
            for _user, task in self.pipeline.active.items():
                if task is None or task.cell_id is None or task.stage != STAGE_TX:
                    continue
                cell = int(task.cell_id)
                n_tx[cell] = n_tx.get(cell, 0) + 1
            rescore = []
            for cell, group in by_cell.items():
                if len(group) <= 1:
                    continue
                n = n_tx.get(cell, 0) + len(group)
                self._cto_share[cell] = float(self.total_bandwidth) / max(n, 1)
                rescore.extend(group)
            if rescore:
                self._bw_override = self._cto_share_bw
                reqs = []
                for user in rescore:
                    cell = assigned[user][1]
                    req = make_request(user)
                    req = (req[0], [cell], req[2], req[3], req[4], req[5], req[6])
                    reqs.append(req)
                picked = mab.select_arms_batch(reqs, advance_slot=False)
                decision_s += float(mab.last_parallel_decision_s)
                for user in rescore:
                    model_idx, cell_id = picked[user]
                    assigned[user] = (int(model_idx), int(cell_id))
        finally:
            self._bw_override = None
            self._cto_share = {}
            self._end_slot_decision_cache()

        mab.last_parallel_decision_s = decision_s
        self._last_parallel_decision_s = decision_s
        user_index = {user: i for i, user in enumerate(self.pipeline.users)}
        for user in self.users:
            if user in assigned:
                model_idx, cell_id = assigned[user]
                cell_rank = cand_cells_dic[user].index(cell_id)
                model_selection_dic[user] = {
                    "model": self.models[model_idx]["name"],
                    "cell_rank": cell_rank,
                }
                cell_dic[user] = cell_id
            else:
                model_idx = next(
                    (i for i, m in enumerate(self.models)
                     if m["name"] == model_selection_dic[user]["model"]), 0)
                cell_rank = 0
            ui = user_index.get(user)
            if ui is not None:
                self.action_freq[
                    ui, model_idx, min(cell_rank, self.top_l_cells - 1)] += 1
        return model_selection_dic, cell_dic


if __name__ == "__main__":
    start_time = time.time()
    config = Config(42)
    cto = CTO(config)
    cto.simulation()
    print("aver info:", cto.average_metrics)
    print("std info:", cto.std_metrics)
    print("action freq info:", cto.action_freq)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
