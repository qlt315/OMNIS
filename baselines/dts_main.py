import numpy as np
import time
from scipy.special import erf
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sys_data.config import Config
import omnis.util as util  # Assuming UtilityFunction is part of util module
from omnis.bcd_loop import run_bcd_slot
import omnis.bcd_loop as bcd_loop
from omnis.assoc_info import (
    init_cell_compute_state,
    update_cell_compute_state,
    expected_gpu_if_join,
    coarse_rank_cells,
)
from omnis.radio_obs import observe_cell_sinr_db, payload_bits
from omnis.task_pipeline import TaskPipeline, task_arrivals
from omnis.sim_loop import run_discrete_simulation
from omnis.queue_agent_mixin import init_task_pipeline, task_dpp_drift


class DTS:
    def __init__(self,config):
        """Initialize system parameters including models, devices, and users."""
        # Basic information
        self.name = "dts"
        self.seed = config.seed
        np.random.seed(self.seed)  # Ensure reproducibility
        random.seed(self.seed)

        # Computation-related parameters
        self.models = config.models
        self.data_size = config.data_size
        self.head_flops = config.head_flops
        self.tail_flops = config.tail_flops

        # User and system environment parameters
        self.users = config.users
        self.user_num = config.user_num
        self.md_params = config.md_params
        self.es_params = config.es_params
        self.time_slot_num = config.time_slot_num

        # Network and transmission parameters
        self.mcs_table = config.mcs_table
        self.available_mcs = config.available_mcs
        self.total_bandwidth = config.total_bandwidth
        self.noise_power_dBm = config.noise_power_dBm
        self.noise_power = config.noise_power

        # Fixed execution costs
        self.fixed_delay = config.fixed_delay
        self.fixed_energy = config.fixed_energy
        self.fixed_energy_weight = config.fixed_energy_weight
        self.delay_constraint_range = config.delay_constraint_range
        self.energy_constraint_range = config.energy_constraint_range
        self.energy_weight_range = config.energy_weight_range

        # Performance metrics
        self.instant_metrics = config.instant_metrics
        self.average_metrics = config.average_metrics
        self.std_metrics = config.std_metrics
        self.est_err = config.est_err
        self.est_err_db = getattr(config, 'est_err_db', 1.0)
        self.sinr_offset_db = float(getattr(config, 'sinr_offset_db', 0.0))
        self.sinr_trace = config.sinr_trace
        self.top_l_cells = config.top_l_cells
        self.num_cells = config.num_cells
        self.assoc_w_radio = float(getattr(config, "assoc_w_radio", 1.0))
        self.assoc_w_compute = float(getattr(config, "assoc_w_compute", 1.0))
        # Std of the per-task accuracy observation noise (0 disables)
        self.acc_noise_std = getattr(config, 'acc_noise_std', 0.0)
        self.action_freq = config.action_freq
        self.decision_time = 0.0
        self.bcd_time = 0.0
        self.bcd_iters = 0.0  # running avg BCD iterations / slot
        self.update_time = 0.0

        # BCD (Block Coordinate Descent) algorithm parameters
        self.bcd_flag = config.bcd_flag
        self.bcd_max_iter = config.bcd_max_iter

        # Contextual decision-making parameters
        self.contexts = config.contexts
        self.action = config.action
        self.context_dim = config.context_dim
        self.action_dim = config.action_dim
        self.kernel = config.kernel
        self.noise = config.noise

        # Bayesian optimization parameters
        self.beta_function = config.beta_function
        self.beta_const_val = config.beta_const_val
        self.optimizers = config.optimizers
        self.utility = util.UtilityFunction(kind="ts",
                                            beta_kind=self.beta_function,
                                            beta_const=self.beta_const_val)

        # Queueing model + Lyapunov framework (journal extension)
        self.slot_duration = getattr(config, 'slot_duration', 1.0)
        self.arrival_rate = getattr(config, 'arrival_rate', {user: 0.7 for user in self.users})
        self.energy_budget = getattr(config, 'energy_budget', {user: 0.45 for user in self.users})
        self.lyapunov_v = getattr(config, 'lyapunov_v', 1.0)
        self.reward_w_acc = getattr(config, 'reward_w_acc', 1.0)
        self.reward_qos_coef = getattr(config, 'reward_qos_coef', 1.5)
        self.dpp_bit_scale = getattr(config, 'dpp_bit_scale', 2.2e4)
        self.dpp_energy_scale = getattr(config, 'dpp_energy_scale', 0.45)
        # Acc/reward learning piggybacks on next-slot uplink (0 = same-slot).
        self.learn_delay_slots = int(getattr(config, 'learn_delay_slots', 1) or 0)
        self._pending_learn = None
        init_task_pipeline(self, config)
        self._last_bandwidth = {}
        self._last_gpu = {}
        self._last_mcs = {}
        # MAB ablations + optional reward prediction-error logging
        self.mab_no_update = bool(getattr(config, 'mab_no_update', False))
        self.mab_freeze_after = int(getattr(config, 'mab_freeze_after', 0) or 0)
        self.log_pred_error = bool(getattr(config, 'log_pred_error', True))
        self._mab_update_slots = 0
        self.pred_err_reward = []
        # MD-visible per-cell compute broadcast (updated after each BCD slot)
        self._cell_compute_state = init_cell_compute_state(
            self.num_cells, self.es_params['freq'], self.user_num)
        # distributed decision_ms = parallel (max-agent)
        self._oh_cache = None
        self._oh_user_base = None
        self._last_parallel_decision_s = 0.0
        self._last_parallel_update_s = 0.0

    def _begin_slot_decision_cache(self):
        self._oh_cache = {}
        self._oh_user_base = {}
        default_bw = self.total_bandwidth / self.user_num
        for user in self.users:
            md = self.md_params[user]
            bw = self._last_bandwidth.get(user, default_bw)
            backlog = float(self.pipeline.tx_queue_len(user))
            p_tx = float(md['trans_power'])
            base = {}
            for model in self.models:
                name = model['name']
                local_d = (self.head_flops[name] * 1e-9
                           / (md['freq'] * md['cores'] * md['flops_per_cycle']))
                local_e = md['power_coeff'] * md['freq'] ** 3 * local_d
                base[name] = (local_d, local_e, payload_bits(self.data_size, name),
                              bw, backlog, p_tx)
            self._oh_user_base[user] = base

    def _gpu_hat_for_association(self, user, cell_id=None):
        if cell_id is not None:
            return expected_gpu_if_join(
                cell_id, self._cell_compute_state,
                self.es_params['freq'], self.user_num)
        return float(self._last_gpu.get(
            user, self.es_params['freq'] / max(self.user_num, 1)))

    def _end_slot_decision_cache(self):
        self._oh_cache = None
        self._oh_user_base = None

    def generate_tasks(self, time_slot):
            """Poisson arrivals. Each task draws its own delay, energy, and weights."""
            from omnis.exog import poisson_arrivals
            return task_arrivals(self, poisson_arrivals(self, time_slot),
                                 time_slot=time_slot)

    def observe_context(self, task_dic, trans_rate_dic):
        """Observe the current context as continuous variables."""
        context_dic = {
            user: {
                "delay_constraint": task_dic[user]["delay_constraint"],
                "energy_constraint": task_dic[user]["energy_constraint"],
                "transmission_rate": trans_rate_dic[user],
                "energy_weight": task_dic[user]["energy_weight"],
                "delay_weight": task_dic[user]["delay_weight"]
            }
            for user in self.users
        }
        return context_dic

    def _users_by_cell(self, cell_dic):
        """Group users by their associated cell index for per-cell ES optimization."""
        by_cell = {}
        for user, cell in cell_dic.items():
            by_cell.setdefault(cell, []).append(user)
        return by_cell

    def _apply_cell_association(self, cell_dic, sinr_db_all_dic):
        """Build linear SNR dict for each user's chosen cell."""
        snr_dic = {}
        for user, cell_idx in cell_dic.items():
            snr_db = sinr_db_all_dic[user][cell_idx]
            snr_dic[user] = 10 ** (snr_db / 10)
        return snr_dic

    def _payload_bits(self, model_name):
        return payload_bits(self.data_size, model_name)

    def _flush_pending_learn(self):
        """Apply piggybacked Acc/reward labels from the previous slot."""
        pend = self._pending_learn
        if pend is None:
            self._last_parallel_update_s = 0.0
            return
        self.update_gp(pend["context"], pend["model_selection"], pend["reward"])
        self._pending_learn = None

    def model_selection(self, context_dic, task_dic, cand_cells_dic, sinr_db_all_dic):
        """Select the best (model, cell_rank) arm for each user using Thompson Sampling.

        Parallel decision: max over per-MD timers (not sequential sum).
        """
        model_selection_dic = {}
        cell_dic = {}
        self._begin_slot_decision_cache()
        user_times = []

        for user_idx, user in enumerate(self.users):
            t_u = time.time()
            top_cells = cand_cells_dic[user]
            locked_cell = self.pipeline.locked_cell(user)
            locked_model = self.pipeline.locked_model(user)
            if locked_cell is not None and locked_model is not None:
                model_idx = next(
                    (i for i, m in enumerate(self.models) if m["name"] == locked_model), 0)
                cell_rank = (top_cells.index(locked_cell)
                             if locked_cell in top_cells else 0)
                self.action_freq[user_idx, model_idx, min(cell_rank, self.top_l_cells - 1)] += 1
                model_selection_dic[user] = {
                    "model": locked_model, "cell_rank": cell_rank}
                cell_dic[user] = locked_cell
                user_times.append(time.time() - t_u)
                continue
            context_m = context_dic[user]
            optimizer_m = self.optimizers[user]
            sinr_db_all = sinr_db_all_dic[user]
            num_joint = len(self.models) * self.top_l_cells
            drift_offsets = np.zeros(num_joint)
            for model_idx, model in enumerate(self.models):
                for cell_rank in range(self.top_l_cells):
                    cell_idx = top_cells[cell_rank]
                    snr_db = sinr_db_all[cell_idx]
                    flat = model_idx * self.top_l_cells + cell_rank
                    mcs_hat = self.forward_sim_mcs(
                        user, snr_db, model['name'], task_dic[user],
                        cell_id=cell_idx)
                    _, _, energy_hat = self.predict_md_overheads(
                        user, None, model['name'], mcs_hat, snr_db=snr_db,
                        cell_id=cell_idx)
                    drift_offsets[flat] = self.dpp_drift(
                        user, model['name'], mcs_hat, energy_hat, snr_db=snr_db,
                        cell_id=cell_idx)
            action_m = optimizer_m.suggest(context_m, self.utility,
                                           score_scale=self.lyapunov_v,
                                           score_offset=drift_offsets)
            selected_model_m = action_m['model']
            selected_cell_rank = action_m['cell_rank']
            cell_id = top_cells[selected_cell_rank]
            self.action_freq[user_idx, selected_model_m, selected_cell_rank] += 1
            model_selection_dic[user] = {
                "model": self.models[selected_model_m]["name"],
                "cell_rank": selected_cell_rank,
            }
            cell_dic[user] = cell_id
            user_times.append(time.time() - t_u)
        self._end_slot_decision_cache()
        self._last_parallel_decision_s = max(user_times) if user_times else 0.0
        return model_selection_dic, cell_dic

    def _mab_allow_update(self):
        if self.mab_no_update:
            return False
        if self.mab_freeze_after > 0 and self._mab_update_slots >= self.mab_freeze_after:
            return False
        return True

    def update_gp(self, context_dic, model_selection_dic, reward_dic):
        """Update the GP model with new observations (unless no_update / freeze).

        Parallel update: max over per-MD timers.
        """
        allow = self._mab_allow_update()
        reward_errs = []
        user_times = []
        for user in self.users:
            t_u = time.time()
            optimizer_m = self.optimizers[user]
            context_m = context_dic[user]
            model_m = model_selection_dic[user]['model']
            action_m = next((index for index, model in enumerate(self.models) if model['name'] == model_m), None)
            action_dic_m = {
                'model': action_m,
                'cell_rank': model_selection_dic[user]['cell_rank'],
            }
            reward_m = reward_dic[user]
            if self.log_pred_error:
                mu = optimizer_m.predict_mean(context_m, action_dic_m)
                if mu is not None:
                    reward_errs.append(abs(float(reward_m) - float(mu)))
            if allow:
                optimizer_m.register(context_m, action_dic_m, reward_m)
            user_times.append(time.time() - t_u)
        if self.log_pred_error:
            self.pred_err_reward.append(
                float(np.mean(reward_errs)) if reward_errs else float("nan"))
        self._mab_update_slots += 1
        self._last_parallel_update_s = max(user_times) if user_times else 0.0

    def predict_md_overheads(self, user, rate_m, model_name, mcs_idx, snr_db=0.0,
                             cell_id=None):
        """Analytic Payload -> {Delay, Energy}; GPU from cell-compute broadcast."""
        del rate_m
        es = self.es_params
        gpu_hat = self._gpu_hat_for_association(user, cell_id=cell_id)
        cache = self._oh_cache
        if cache is not None and self._oh_user_base is not None:
            key = (user, model_name, int(mcs_idx), float(snr_db),
                   -1 if cell_id is None else int(cell_id))
            hit = cache.get(key)
            if hit is not None:
                return hit
            local_d, local_e, payload, bw, backlog, p_tx = (
                self._oh_user_base[user][model_name])
            edge_d = (self.tail_flops[model_name] * 1e-9
                      / (gpu_hat * es['cores'] * es['flops_per_cycle']))
            edge_e = es['power_coeff'] * gpu_hat ** 3 * edge_d
            # Delay/energy: clamped delivery SE (avoid 1/1e-12 blow-ups).
            # Queue drain elsewhere still uses uncapped goodput_se.
            se_eff = self.mcs_table.delay_se(model_name, mcs_idx, snr_db)
            rate_hat = max(bw * se_eff, 1e-12)
            trans_delay = payload / max(rate_hat, 1e-12)
            trans_energy = p_tx * trans_delay
            service_delay = local_d + trans_delay + edge_d
            q_j = float(self.pipeline.jobs_ahead(user, cell_id=cell_id))
            sojourn_delay = service_delay + q_j * edge_d
            total_energy = local_e + trans_energy + edge_e
            out = (service_delay, sojourn_delay, total_energy)
            cache[key] = out
            return out

        md = self.md_params[user]
        es = self.es_params

        head_flops = self.head_flops[model_name]
        local_delay = head_flops * 1e-9 / (md['freq'] * md['cores'] * md['flops_per_cycle'])
        local_energy = md['power_coeff'] * md['freq'] ** 3 * local_delay

        bandwidth_hat = self._last_bandwidth.get(user, self.total_bandwidth / self.user_num)
        se_eff = self.mcs_table.delay_se(model_name, mcs_idx, snr_db)
        rate_hat = max(bandwidth_hat * se_eff, 1e-12)  # [bit/s]
        bits = self._payload_bits(model_name)
        trans_delay = bits / rate_hat
        trans_energy = md['trans_power'] * trans_delay

        gpu_hat = self._gpu_hat_for_association(user, cell_id=cell_id)
        tail_flops = self.tail_flops[model_name]
        edge_delay = tail_flops * 1e-9 / (
            max(gpu_hat, 1e-12) * es['cores'] * es['flops_per_cycle'])
        edge_energy = es['power_coeff'] * gpu_hat ** 3 * edge_delay

        service_delay = local_delay + trans_delay + edge_delay
        q_j = float(self.pipeline.jobs_ahead(user, cell_id=cell_id))
        sojourn_delay = service_delay + q_j * edge_delay
        total_energy = local_energy + trans_energy + edge_energy
        return service_delay, sojourn_delay, total_energy


    def dpp_drift(self, user, model_name, mcs_idx, energy_hat, snr_db=0.0, cell_id=None):
        return task_dpp_drift(
            self, user, model_name, mcs_idx, energy_hat, snr_db=snr_db, cell_id=cell_id)

    def forward_sim_mcs(self, user, snr_db, model_name, task_u, cell_id=None):
        """ILLA MCS forward sim: BLER/SE + QoS only (no Acc-table scoring)."""
        bler_t = getattr(self, 'bler_target', self.mcs_table.bler_target)
        es = self.es_params
        gpu_hat = self._gpu_hat_for_association(user, cell_id=cell_id)
        if self._oh_user_base is not None:
            local_d, local_e, payload, bw, _backlog, p_tx = (
                self._oh_user_base[user][model_name])
            edge_d = (self.tail_flops[model_name] * 1e-9
                      / (gpu_hat * es['cores'] * es['flops_per_cycle']))
            edge_e = es['power_coeff'] * gpu_hat ** 3 * edge_d
            bler, goodput = self.mcs_table.bler_goodput_all_mcs(model_name, snr_db)
            mcs_idx = np.asarray(self.available_mcs, dtype=int)
            se = self.mcs_table._se_arr
            rates = bw * np.maximum(goodput, 1e-12)
            trans_d = payload / rates
            service = local_d + trans_d + edge_d
            energy = local_e + p_tx * trans_d + edge_e
            d_c = task_u['delay_constraint']
            e_c = task_u['energy_constraint']
            feas_m = (service <= d_c) & (energy <= e_c)
            if np.any(feas_m):
                under = feas_m & (bler <= bler_t)
                mask = under if np.any(under) else feas_m
                best_j = int(np.argmax(np.where(mask, se, -np.inf)))
                return int(mcs_idx[best_j])
            score = (task_u['delay_weight'] * erf(d_c - service)
                     + task_u['energy_weight'] * erf(e_c - energy))
            return int(mcs_idx[int(np.argmax(score))])
        feas = []
        best_infeas, best_infeas_score = None, -np.inf
        for mcs in self.available_mcs:
            service_hat, _, energy_hat = self.predict_md_overheads(
                user, None, model_name, mcs, snr_db=snr_db, cell_id=cell_id)
            if (service_hat <= task_u['delay_constraint']
                    and energy_hat <= task_u['energy_constraint']):
                bler = self.mcs_table.bler(model_name, mcs, snr_db)
                feas.append((mcs, bler, self.mcs_table.se[mcs]))
            else:
                score = (task_u['delay_weight'] * erf(task_u['delay_constraint'] - service_hat)
                         + task_u['energy_weight'] * erf(task_u['energy_constraint'] - energy_hat))
                if score > best_infeas_score:
                    best_infeas, best_infeas_score = mcs, score
        if feas:
            under = [t for t in feas if t[1] <= bler_t]
            pool = under if under else feas
            return max(pool, key=lambda t: t[2])[0]
        return best_infeas


    def get_reward(self, task_dic, acc_dic, total_overhead_dic):
        """Utility: w_acc * accuracy + qos_coef * delay/energy erf terms."""
        w_acc = getattr(self, "reward_w_acc", 1.0)
        qos = getattr(self, "reward_qos_coef", 1.5)
        reward_dic = {}
        for user in acc_dic.keys():
            reward_dic[user] = (
                w_acc * acc_dic[user]
                + qos * task_dic[user]["delay_weight"] * erf(
                    task_dic[user]["delay_constraint"] - total_overhead_dic[user]["delay"])
                + qos * task_dic[user]["energy_weight"] * erf(
                    task_dic[user]["energy_constraint"] - total_overhead_dic[user]["energy"]))
        return reward_dic

    def get_trans_rate(self, time_slot):
        """Per-UE radio observation + coarse Top-L association candidates.

        Returns snr_best_est, trans_rate, cand_cells, sinr_est_db_all, sinr_true_db_all.
        Est CSI → association / decisions / BCD MCS; true CSI → Acc / goodput env.
        """
        trans_rate_dic = {}
        snr_dic = {}
        cand_cells_dic = {}
        sinr_est_db_all_dic = {}
        sinr_true_db_all_dic = {}

        for user_idx, user in enumerate(self.users):
            true_db, est_db = observe_cell_sinr_db(
                self.sinr_trace, time_slot, user_idx,
                sinr_offset_db=self.sinr_offset_db,
                est_err_db=self.est_err_db,
                seed=self.seed)
            top_cells = coarse_rank_cells(
                est_db,
                getattr(self, "_cell_compute_state", None),
                self.top_l_cells,
                self.es_params["freq"],
                self.user_num,
                w_radio=getattr(self, "assoc_w_radio", 1.0),
                w_compute=getattr(self, "assoc_w_compute", 1.0),
            )
            cand_cells_dic[user] = top_cells
            sinr_est_db_all_dic[user] = est_db
            sinr_true_db_all_dic[user] = true_db
            best_cell = top_cells[0]
            best_snr_linear = 10 ** (est_db[best_cell] / 10)
            snr_dic[user] = best_snr_linear
            trans_rate_dic[user] = best_snr_linear

        return (snr_dic, trans_rate_dic, cand_cells_dic,
                sinr_est_db_all_dic, sinr_true_db_all_dic)


    def _goodput_se(self, user, model_name, mcs_idx, snr_dic):
        """Effective SE after TB erasures [bit/s/Hz] (queue service)."""
        snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
        return max(self.mcs_table.goodput_se(model_name, mcs_idx, snr_db), 0.0)

    def _delay_se(self, user, model_name, mcs_idx, snr_dic):
        """Clamped delivery SE for delay/energy / BW weights."""
        snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
        return self.mcs_table.delay_se(model_name, mcs_idx, snr_db)

    def allocate_bandwidth(self, task_dic, model_selection_dic, trans_rate_dic, phy_choice_dic,
                           users=None, snr_dic=None):
        """Allocate bandwidth within one cell's pool (delay-SE-weighted)."""
        users = self.users if users is None else users
        d_prime_dic = {}
        for user in users:
            chosen_model_m = model_selection_dic[user]["model"]
            p_m = self.md_params[user]['trans_power']
            omega_m_t = task_dic[user]['delay_weight']
            omega_m_e = task_dic[user]['energy_weight']
            q_n = self.pipeline.composite_backlog(user) / self.dpp_task_scale
            se_eff = (self._delay_se(user, chosen_model_m, phy_choice_dic[user], snr_dic)
                      if snr_dic is not None else max(self.mcs_table.se[phy_choice_dic[user]], 1e-12))
            bits = self.pipeline.uplink_residual_bits(user)
            if bits <= 1e-9:
                bits = self._payload_bits(chosen_model_m)
            d_prime_dic[user] = ((1 + q_n) * bits
                                 * (omega_m_t + p_m * omega_m_e) / se_eff)

        total_sqrt_d_prime = sum(np.sqrt(d) for d in d_prime_dic.values())
        if total_sqrt_d_prime <= 1e-18:
            n = max(len(users), 1)
            return {user: self.total_bandwidth / n for user in users}
        return {
            user: self.total_bandwidth * np.sqrt(d_prime_dic[user]) / total_sqrt_d_prime
            for user in users
        }

    def allocate_bandwidth_all_cells(self, task_dic, model_selection_dic, trans_rate_dic,
                                     phy_choice_dic, cell_dic, snr_dic=None):
        """Per-cell bandwidth allocation; cells independent after association."""
        return bcd_loop.allocate_bandwidth_all_cells(
            self, task_dic, model_selection_dic, trans_rate_dic, phy_choice_dic,
            cell_dic, snr_dic=snr_dic)

    def gpu_resource_allocation(self, task_dic, model_selection_dic, users=None):
        """FIFO edge service: computing-queue head gets the full GPU pool."""
        users = self.users if users is None else users
        gpu_allocation_dict = {user: 0.0 for user in users}
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            n = max(len(users), 1)
            return {u: self.es_params["freq"] / n for u in users}
        f_tot = float(self.es_params["freq"])
        claimed = set()
        for cell in range(self.num_cells):
            q = pipe.edge_q[cell]
            if not q:
                continue
            head = q[0]
            if head.user in users and head.user not in claimed:
                gpu_allocation_dict[head.user] = f_tot
                claimed.add(head.user)
        return gpu_allocation_dict


    def gpu_resource_allocation_all_cells(self, task_dic, model_selection_dic, cell_dic):
        """Per-cell GPU allocation; cells independent after association."""
        return bcd_loop.gpu_resource_allocation_all_cells(
            self, task_dic, model_selection_dic, cell_dic)

    def mcs_selection(self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                      local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic, users=None):
        """ILLA BLER filter + drift/QoS score (no Acc-table term). Learns reward via GP."""
        users = self.users if users is None else users
        bler_t = getattr(self, 'bler_target', self.mcs_table.bler_target)
        mcs_dic = {}
        edge_overhead_dic = self.get_edge_overhead(model_selection_dic, gpu_allocation_dic)
        for user in users:
            model_name_u = model_selection_dic[user]["model"]
            snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
            under, over, best_infeas, best_infeas_score = [], [], None, -np.inf
            for mcs in self.available_mcs:
                temp_mcs_dic = {user: mcs}
                trans_overhead_dic = self.get_trans_overhead(
                    {user: trans_rate_dic[user]}, {user: model_selection_dic[user]},
                    {user: bandwidth_allocation_dic[user]}, temp_mcs_dic,
                    snr_dic={user: snr_dic[user]})
                service_delay = (local_overhead_dic[user]['delay'] + edge_overhead_dic[user]['delay'] +
                                 trans_overhead_dic[user]['delay'])
                total_energy = (local_overhead_dic[user]['energy'] + edge_overhead_dic[user]['energy'] +
                                trans_overhead_dic[user]['energy'])
                drift = self.dpp_drift(user, model_name_u, mcs, total_energy, snr_db=snr_db)
                if (service_delay <= task_dic[user]['delay_constraint']
                        and total_energy <= task_dic[user]['energy_constraint']):
                    score = drift
                    bler = self.mcs_table.bler(model_name_u, mcs, snr_db)
                    entry = (mcs, score, bler, self.mcs_table.se[mcs])
                    if bler <= bler_t:
                        under.append(entry)
                    else:
                        over.append(entry)
                else:
                    score = (self.lyapunov_v * (
                             task_dic[user]['delay_weight']
                             * erf(task_dic[user]['delay_constraint'] - service_delay)
                             + task_dic[user]['energy_weight']
                             * erf(task_dic[user]['energy_constraint'] - total_energy))
                             + drift)
                    if score > best_infeas_score:
                        best_infeas, best_infeas_score = mcs, score
            pool = under if under else over
            if pool:
                mcs_dic[user] = max(pool, key=lambda t: (t[1], t[3]))[0]
            else:
                mcs_dic[user] = best_infeas
        return mcs_dic


    def mcs_selection_all_cells(self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                                local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic,
                                cell_dic):
        """Per-cell MCS selection; cells independent after association."""
        return bcd_loop.mcs_selection_all_cells(
            self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
            local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic, cell_dic)

    def get_accuracy(self, snr_dic, mcs_dic, model_selection_dic):
        """Environment Acc realization from the PHY Acc table (not for decisions)."""
        acc_dic = {}
        for user in snr_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
            acc_dic[user] = self.mcs_table.accuracy(chosen_model_m, mcs_dic[user], snr_db)
        return acc_dic

    def get_local_overhead(self, model_selection_dic):
        """Local residual delay/energy (0 once the active task has left local)."""
        from omnis.task_pipeline import STAGE_LOCAL

        local_overhead_dic = {}
        for user in model_selection_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            head_flops = self.head_flops[chosen_model_m]
            flops_per_cycle = self.md_params[user]['flops_per_cycle']
            num_cores = self.md_params[user]['cores']
            gpu_freq = self.md_params[user]['freq']
            full_delay = head_flops * 1e-9 / (gpu_freq * num_cores * flops_per_cycle)
            at = self.pipeline.active.get(user)
            if at is None:
                local_delay = full_delay
            elif at.stage == STAGE_LOCAL:
                local_delay = float(at.residual)
            else:
                local_delay = 0.0
            local_energy = self.md_params[user]['power_coeff'] * gpu_freq ** 3 * local_delay
            local_overhead_dic[user] = {
                "delay": local_delay,
                "energy": local_energy
            }
        return local_overhead_dic


    def get_trans_overhead(self, trans_rate_dic, model_selection_dic, bandwidth_allocation_dic,
                           mcs_dic, snr_dic=None):
        """Transmission delay/energy via clamped delivery SE (stable QoS)."""
        trans_overhead_dic = {}
        for user in model_selection_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            bandwidth_m = max(float(bandwidth_allocation_dic[user]), 1e-12)
            data_size_m = self.pipeline.uplink_residual_bits(user)
            if data_size_m <= 1e-9:
                data_size_m = self._payload_bits(chosen_model_m)
            if snr_dic is not None:
                snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
                se_eff = self.mcs_table.delay_se(
                    chosen_model_m, mcs_dic[user], snr_db)
            else:
                se_eff = max(self.mcs_table.se[mcs_dic[user]], 1e-12)
            trans_delay = data_size_m / (bandwidth_m * se_eff)
            trans_energy = self.md_params[user]['trans_power'] * trans_delay
            trans_overhead_dic[user] = {"delay": trans_delay, "energy": trans_energy}
        return trans_overhead_dic


    def get_edge_overhead(self, model_selection_dic, gpu_allocation_dic):
        """Calculate edge processing overhead, including delay and energy consumption, for selected models."""

        edge_overhead_dic = {}  # Dictionary to store delay and energy consumption for each user

        for user in model_selection_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]  # Get the selected model for this user
            # Extract relevant parameters
            tail_flops_m = self.tail_flops[chosen_model_m]  # FLOPs of the tail model
            flops_per_cycle_m = self.es_params['flops_per_cycle']  # FLOPs per cycle for this user
            num_cores_m = self.es_params['cores']  # Number of cores for this user
            gpu_freq_m = max(float(gpu_allocation_dic.get(user, 0.0)), 1e-12)
            if gpu_freq_m <= 1e-12:
                gpu_freq_m = float(self.es_params['freq'])

            # Compute edge processing delay
            edge_delay = tail_flops_m * 1e-9 / (gpu_freq_m * num_cores_m * flops_per_cycle_m)

            # Compute edge energy consumption with the ES power coefficient
            edge_energy = self.es_params['power_coeff'] * gpu_freq_m ** 3 * edge_delay

            # Store delay and energy in the result dictionary for the user
            edge_overhead_dic[user] = {
                "delay": edge_delay,
                "energy": edge_energy
            }

        return edge_overhead_dic  # Return dictionary with local overhead for all users

    def get_total_overhead(self, local_overhead_dic, trans_overhead_dic, edge_overhead_dic,
                           queue_wait_dic=None):
        total_overhead_dic = {}
        for user in self.users:
            queue_wait = 0.0 if queue_wait_dic is None else queue_wait_dic[user]
            total_delay = queue_wait + local_overhead_dic[user]['delay'] + trans_overhead_dic[user]['delay'] + \
                          edge_overhead_dic[user]['delay']
            total_energy = local_overhead_dic[user]['energy'] + trans_overhead_dic[user]['energy'] + \
                           edge_overhead_dic[user]['energy']
            total_overhead_dic[user] = {
                "delay": total_delay,
                "energy": total_energy
            }
        return total_overhead_dic

    def moving_average(self, data, window_size):
        """Apply moving average to smooth the data."""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def show_convergence(self):
        """Smooth and plot the rewards, latency, energy, accuracy, is_vio, and vio_degree for each user."""
        window_size = 20
        slots = np.arange(1, self.time_slot_num + 1)  # X-axis: Time slot index

        # Apply moving average to smooth the rewards for each user
        for user in self.users:
            self.instant_metrics[user]['reward'] = self.moving_average(self.instant_metrics[user]['reward'],
                                                                       window_size)

        # Set up a 3x2 grid for plotting
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))

        # Plot the rewards for each user over time slots
        for user in self.users:
            axes[0, 0].plot(self.instant_metrics[user]['reward'], label=f'{user}')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_xlabel('Time Slot')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot latency for all users
        for user in self.users:
            axes[0, 1].plot(slots, self.instant_metrics[user]["delay"], label=f"{user}")
        axes[0, 1].set_ylabel("Latency [s]")
        axes[0, 1].set_xlabel("Time Slot")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot energy consumption for all users
        for user in self.users:
            axes[1, 0].plot(slots, self.instant_metrics[user]["energy"], label=f"{user}")
        axes[1, 0].set_ylabel("Energy [J]")
        axes[1, 0].set_xlabel("Time Slot")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot accuracy for all users
        for user in self.users:
            axes[1, 1].plot(slots, self.instant_metrics[user]["accuracy"], label=f"{user}")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_xlabel("Time Slot")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Plot is_vio for all users
        for user in self.users:
            axes[2, 0].plot(slots, self.instant_metrics[user]["is_vio"], label=f"{user}")
        axes[2, 0].set_ylabel("Is Violation ")
        axes[2, 0].set_xlabel("Time Slot")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Plot vio_degree for all users
        for user in self.users:
            axes[2, 1].plot(slots, self.instant_metrics[user]["vio_degree"], label=f"{user}")
        axes[2, 1].set_ylabel("Violation Excess")
        axes[2, 1].set_xlabel("Time Slot")
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def get_average_and_std_metrics(self):
        """Task QoS over completions; queues over all slots."""
        from omnis.queue_agent_mixin import get_average_and_std_metrics as _avg
        _avg(self)


    def get_instant_metrics(self, task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic=None):
        for user in self.users:
            self.instant_metrics[user]["delay"].append(total_overhead_dic[user]['delay'])
            self.instant_metrics[user]["energy"].append(total_overhead_dic[user]['energy'])
            self.instant_metrics[user]["accuracy"].append(acc_dic[user])
            self.instant_metrics[user]["reward"].append(reward_dic[user])
            reward_dic[user] = float(reward_dic[user])
            if queue_info_dic is not None:
                for key in ("backlog", "energy_queue", "arrivals", "served"):
                    self.instant_metrics[user][key].append(queue_info_dic[user][key])

            # Get the current user's constraint values and weights
            delay = total_overhead_dic[user]['delay']
            energy = total_overhead_dic[user]['energy']
            delay_constraint = task_dic[user]["delay_constraint"]
            energy_constraint = task_dic[user]["energy_constraint"]
            delay_weight = task_dic[user]["delay_weight"]
            energy_weight = task_dic[user]["energy_weight"]

            # Initialize violation degree and violation flag
            vio_degree = 0
            is_vio = 0

            # Check if any constraints are violated
            if delay > delay_constraint or energy > energy_constraint:
                is_vio = 1  # Set violation flag
                # If delay constraint is violated, calculate the violation degree
                if delay > delay_constraint:
                    vio_degree += delay_weight * (delay - delay_constraint)
                # If energy constraint is violated, calculate the violation degree
                if energy > energy_constraint:
                    vio_degree += energy_weight * (energy - energy_constraint)

            # Update the results
            self.instant_metrics[user]["is_vio"].append(is_vio)
            self.instant_metrics[user]["vio_degree"].append(vio_degree)

        # print("reward dic:", reward_dic)
        return self.instant_metrics

    def simulation(self):
        """Discrete tasks + sticky BS + compute queue (shared loop)."""
        run_discrete_simulation(self)



if __name__ == "__main__":
    start_time = time.time()  # Record start time
    seed = 42
    config = Config(seed)
    dts = DTS(config)
    dts.simulation()
    print("aver info:", dts.average_metrics)
    print("std info:", dts.std_metrics)
    print("action freq info:", dts.action_freq)
    end_time = time.time()  # Record end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    dts.show_convergence()

