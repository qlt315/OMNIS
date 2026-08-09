import numpy as np
from scipy.special import erf
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sys_data.config import Config
from omnis.causal_scm import CausalSCM
from omnis.causal_bandit import CausalMAB
from omnis.bcd_loop import run_bcd_slot
import omnis.bcd_loop as bcd_loop
import time

class OMNIS:
    def __init__(self, config):
        """Initialize system parameters including models, devices, and users."""
        # Basic information
        self.name = "omnis"
        self.seed = config.seed
        np.random.seed(self.seed)  # Ensure reproducibility
        random.seed(self.seed)  # Seed the Python RNG as well (init coding rates, exploration)

        # Computation-related parameters
        self.models = config.models
        self.data_size = config.data_size
        self.head_flops = config.head_flops
        self.tail_flops = config.tail_flops

        # User and environment-related parameters
        self.users = config.users
        self.user_num = config.user_num
        self.md_params = config.md_params
        self.es_params = config.es_params
        self.time_slot_num = config.time_slot_num

        # Network and transmission-related parameters
        self.total_bandwidth = config.total_bandwidth
        self.noise_power_dBm = config.noise_power_dBm
        self.noise_power = config.noise_power

        # PHY layer: Sionna-generated MCS tables (journal version)
        self.mcs_table = config.mcs_table
        self.available_mcs = config.available_mcs

        # Fixed overhead costs for task execution
        self.fixed_delay = config.fixed_delay
        self.fixed_energy = config.fixed_energy
        self.fixed_energy_weight = config.fixed_energy_weight

        # Performance metrics
        self.instant_metrics = config.instant_metrics
        self.average_metrics = config.average_metrics
        self.std_metrics = config.std_metrics

        # Wall-clock: MD decision + ES BCD + learning update [s]
        self.decision_time = 0.0
        self.bcd_time = 0.0
        self.bcd_iters = 0.0  # running avg BCD iterations / slot
        self.update_time = 0.0
        self.est_err = config.est_err
        self.est_err_db = getattr(config, 'est_err_db', 1.0)
        self.sinr_offset_db = float(getattr(config, 'sinr_offset_db', 0.0))
        self.sinr_trace = config.sinr_trace
        self.top_l_cells = config.top_l_cells
        self.num_cells = config.num_cells
        self.action_freq = config.action_freq

        # BCD (Block Coordinate Descent) algorithm-related parameters
        self.bcd_flag = config.bcd_flag
        self.bcd_max_iter = config.bcd_max_iter

        # Decision-making and optimization-related parameters
        self.contexts = config.contexts
        self.action = config.action
        self.context_dim = config.context_dim
        self.action_dim = config.action_dim
        self.kernel = config.kernel
        self.noise = config.noise
        self.beta_function = config.beta_function
        self.beta_const_val = config.beta_const_val
        self.optimizers = config.optimizers
        self.utility = config.utility

        # Queueing model + Lyapunov framework (journal extension)
        self.slot_duration = getattr(config, 'slot_duration', 1.0)
        self.arrival_rate = getattr(config, 'arrival_rate', {user: 0.7 for user in self.users})
        self.energy_budget = getattr(config, 'energy_budget', {user: 0.45 for user in self.users})
        self.lyapunov_v = getattr(config, 'lyapunov_v', 1.0)
        self.reward_w_acc = getattr(config, 'reward_w_acc', 1.0)
        self.reward_qos_coef = getattr(config, 'reward_qos_coef', 1.5)
        self.dpp_bit_scale = getattr(config, 'dpp_bit_scale', 2.2e4)
        self.dpp_energy_scale = getattr(config, 'dpp_energy_scale', 0.45)
        # Per-user uplink bit queue Q_m and virtual energy queue Z_m
        self.backlog = {user: 0.0 for user in self.users}
        self.energy_queue = {user: 0.0 for user in self.users}

        # Causal MAB (journal extension)
        self.algo = getattr(config, 'algo', 'ucb')
        self.acc_noise_std = getattr(config, 'acc_noise_std', 0.0)
        if self.algo == 'causal':
            self.scm = CausalSCM(
                models=self.models,
                data_size=self.data_size,
                mcs_table=self.mcs_table,
                prior_snr_step=config.causal_prior_snr_step,
                build_prior=False,
            )
            noise_var = max(self.acc_noise_std ** 2, 1e-6)
            self.causal_mab = CausalMAB(
                scm=self.scm,
                length_scales=config.causal_gp_length_scales,
                signal_var=config.causal_gp_signal_var,
                noise_var=noise_var,
                beta=getattr(config, 'causal_beta', self.beta_const_val),
                penalty_gain=self.reward_qos_coef,
                acquisition=config.causal_acq,
                use_prior=False,
                shared=config.causal_shared,
                lyapunov_v=self.lyapunov_v,
                drift_gain=getattr(config, 'causal_drift_gain', 1.0),
                w_acc=self.reward_w_acc,
                init_random=getattr(config, 'causal_init_random', 20),
                empty_prior_std=getattr(config, 'causal_empty_prior_std', 1.0),
                explore_slots=getattr(config, 'causal_explore_slots', 20),
                max_obs=getattr(config, 'causal_gp_max_obs', 600),
            )
        # MAB ablations + prediction-error logging (Causal residual GP / UCB reward GP)
        self.mab_no_update = bool(getattr(config, 'mab_no_update', False))
        self.mab_freeze_after = int(getattr(config, 'mab_freeze_after', 0) or 0)
        self.log_pred_error = bool(getattr(config, 'log_pred_error', True))
        self._mab_update_slots = 0
        self.pred_err_prior = []   # mean |acc - prior| per slot (causal)
        self.pred_err_post = []    # mean |acc - posterior| per slot (causal)
        self.pred_err_reward = []  # mean |reward - GP mean| per slot (ucb)
        # Persistence-based prediction of the ES allocation (last observed values)
        self._last_bandwidth = {}
        self._last_gpu = {}
        self._last_mcs = {}
        # Per-slot decision caches: analytic overheads for (user, model, mcs, snr).
        # distributed decision_ms = parallel (max-agent).
        self._oh_cache = None
        self._oh_user_base = None
        self._last_parallel_decision_s = 0.0
        self._last_parallel_update_s = 0.0

    def _begin_slot_decision_cache(self):
        """Reset per-slot caches for MD analytic overhead prediction."""
        self._oh_cache = {}
        self._oh_user_base = {}
        default_bw = self.total_bandwidth / self.user_num
        default_gpu = self.es_params['freq'] / self.user_num
        es = self.es_params
        for user in self.users:
            md = self.md_params[user]
            bw = self._last_bandwidth.get(user, default_bw)
            gpu = self._last_gpu.get(user, default_gpu)
            backlog = float(self.backlog[user])
            p_tx = float(md['trans_power'])
            base = {}
            for model in self.models:
                name = model['name']
                local_d = (self.head_flops[name] * 1e-9
                           / (md['freq'] * md['cores'] * md['flops_per_cycle']))
                local_e = md['power_coeff'] * md['freq'] ** 3 * local_d
                edge_d = (self.tail_flops[name] * 1e-9
                          / (gpu * es['cores'] * es['flops_per_cycle']))
                edge_e = es['power_coeff'] * gpu ** 3 * edge_d
                base[name] = (local_d, local_e, edge_d, edge_e,
                              float(self.data_size[name]), bw, backlog, p_tx)
            self._oh_user_base[user] = base

    def _end_slot_decision_cache(self):
        self._oh_cache = None
        self._oh_user_base = None

    def generate_tasks(self, time_slot):
            """Dynamically adjust delay and energy constraints while keeping the base values fixed.
            Also draws the Poisson task arrivals for this slot (queueing model)."""

            task_dic = {
                user: {
                    "delay_constraint": self.fixed_delay[user] + np.random.uniform(-0.001, 0.001),
                    "energy_constraint": self.fixed_energy[user] + np.random.uniform(-0.001, 0.001),
                    "energy_weight": self.fixed_energy_weight[user] + np.random.uniform(-0.001, 0.001),
                    "n_arrivals": np.random.poisson(self.arrival_rate[user]),
                }
                for user in self.users
            }

            for user in self.users:
                task_dic[user]["delay_weight"] = 1 - task_dic[user]["energy_weight"]

            return task_dic

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
        for user in self.users:
            cell_idx = cell_dic[user]
            snr_db = sinr_db_all_dic[user][cell_idx]
            snr_dic[user] = 10 ** (snr_db / 10)
        return snr_dic

    def model_selection(self, context_dic, task_dic, cand_cells_dic, sinr_db_all_dic):
        """Select the best (model, cell_rank) arm for each user using GP-UCB.

        The acquisition value is rescaled by the Lyapunov weight V and the
        analytic per-arm drift term is added before the argmax.

        Timing: per-user work is embarrassingly parallel across MDs;
        ``_last_parallel_decision_s`` = max over users (not the sequential sum).
        """
        model_selection_dic = {}
        cell_dic = {}
        self._begin_slot_decision_cache()
        user_times = []

        for user_idx, user in enumerate(self.users):
            t_u = time.time()
            context_m = context_dic[user]
            optimizer_m = self.optimizers[user]
            top_cells = cand_cells_dic[user]
            sinr_db_all = sinr_db_all_dic[user]
            num_joint = len(self.models) * self.top_l_cells
            drift_offsets = np.zeros(num_joint)
            for model_idx, model in enumerate(self.models):
                for cell_rank in range(self.top_l_cells):
                    cell_idx = top_cells[cell_rank]
                    snr_db = sinr_db_all[cell_idx]
                    flat = model_idx * self.top_l_cells + cell_rank
                    mcs_hat = self.forward_sim_mcs(user, snr_db, model['name'], task_dic[user])
                    _, _, energy_hat = self.predict_md_overheads(
                        user, None, model['name'], mcs_hat, snr_db=snr_db)
                    drift_offsets[flat] = self.dpp_drift(
                        user, model['name'], mcs_hat, energy_hat, snr_db=snr_db)
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
        """Whether this slot should register a new GP observation."""
        if self.mab_no_update:
            return False
        if self.mab_freeze_after > 0 and self._mab_update_slots >= self.mab_freeze_after:
            return False
        return True

    def update_gp(self, context_dic, model_selection_dic, reward_dic):
        """Update the GP model with new observations (unless no_update / freeze).

        Per-MD GPs update independently → ``_last_parallel_update_s`` = max_u.
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

    def model_selection_causal(self, task_dic, cand_cells_dic, sinr_db_all_dic, trans_rate_dic):
        """Select the (model, cell) branch for each MD with the causal bandit.

        Precomputes per-user overhead bases once per slot; MCS/arm scoring
        hits ``_oh_cache``. Parallel decision time comes from CausalMAB.
        """
        del trans_rate_dic  # association SINR comes from cand cells
        self._begin_slot_decision_cache()
        requests = []
        for user in self.users:

            def predict_overheads(model_name, mcs_idx, snr_db=0.0, user=user):
                # Per-arm SINR (same cell as GP features), not best-cell persistence.
                return self.predict_md_overheads(
                    user, None, model_name, mcs_idx, snr_db=snr_db)

            def drift_score(model_name, mcs_idx, energy_hat, snr_db=0.0, user=user):
                return self.dpp_drift(user, model_name, mcs_idx, energy_hat, snr_db=snr_db)

            requests.append((
                user, cand_cells_dic[user], sinr_db_all_dic[user],
                task_dic[user], predict_overheads, drift_score,
                self._oh_user_base[user]))

        selected_dic = self.causal_mab.select_arms_batch(requests)
        self._last_parallel_decision_s = float(
            self.causal_mab.last_parallel_decision_s)
        self._end_slot_decision_cache()
        model_selection_dic = {}
        cell_dic = {}
        for user_idx, user in enumerate(self.users):
            selected_model_m, cell_id = selected_dic[user]
            cell_rank = cand_cells_dic[user].index(cell_id)
            self.action_freq[user_idx, selected_model_m, cell_rank] += 1
            model_selection_dic[user] = {"model": self.models[selected_model_m]["name"]}
            cell_dic[user] = cell_id
        return model_selection_dic, cell_dic

    def predict_md_overheads(self, user, rate_m, model_name, mcs_idx, snr_db=0.0):
        """Analytic causal chain Payload -> {Delay, Energy} for an arm candidate,
        using a persistence prediction of the ES bandwidth/GPU allocation.

        Transmission uses goodput SE = (1-BLER)*η so failed TBs are erasures
        (classmate Sionna semantics), not residual-BER corruption.

        Returns (service_delay, sojourn_delay, energy).
        When a slot decision cache is active, local/edge terms are reused and
        (user, model, mcs, snr) results are memoized.
        """
        del rate_m
        cache = self._oh_cache
        if cache is not None and self._oh_user_base is not None:
            key = (user, model_name, int(mcs_idx), float(snr_db))
            hit = cache.get(key)
            if hit is not None:
                return hit
            local_d, local_e, edge_d, edge_e, payload, bw, backlog, p_tx = (
                self._oh_user_base[user][model_name])
            se_eff = self.mcs_table.goodput_se(model_name, mcs_idx, snr_db)
            rate_hat = bw * max(se_eff, 1e-12)
            trans_delay = payload / rate_hat
            queue_delay = backlog / rate_hat
            trans_energy = p_tx * trans_delay
            service_delay = local_d + trans_delay + edge_d
            sojourn_delay = service_delay + queue_delay
            total_energy = local_e + trans_energy + edge_e
            out = (service_delay, sojourn_delay, total_energy)
            cache[key] = out
            return out

        md = self.md_params[user]

        head_flops = self.head_flops[model_name]
        local_delay = head_flops * 1e-9 / (md['freq'] * md['cores'] * md['flops_per_cycle'])
        local_energy = md['power_coeff'] * md['freq'] ** 3 * local_delay

        bandwidth_hat = self._last_bandwidth.get(user, self.total_bandwidth / self.user_num)
        se_eff = self.mcs_table.goodput_se(model_name, mcs_idx, snr_db)
        rate_hat = bandwidth_hat * max(se_eff, 1e-12)  # [bit/s]
        trans_delay = self.data_size[model_name] / rate_hat
        queue_delay = self.backlog[user] / rate_hat
        trans_energy = md['trans_power'] * trans_delay

        gpu_hat = self._last_gpu.get(user, self.es_params['freq'] / self.user_num)
        tail_flops = self.tail_flops[model_name]
        edge_delay = tail_flops * 1e-9 / (
            gpu_hat * self.es_params['cores'] * self.es_params['flops_per_cycle'])
        edge_energy = self.es_params['power_coeff'] * gpu_hat ** 3 * edge_delay

        service_delay = local_delay + trans_delay + edge_delay
        sojourn_delay = service_delay + queue_delay
        total_energy = local_energy + trans_energy + edge_energy
        return service_delay, sojourn_delay, total_energy

    def dpp_drift(self, user, model_name, mcs_idx, energy_hat, snr_db=0.0):
        """Analytic Lyapunov drift term for an arm candidate (normalized to O(1)).

        Service uses goodput SE so queue draining reflects TB erasures."""
        bandwidth_hat = self._last_bandwidth.get(user, self.total_bandwidth / self.user_num)
        se_eff = self.mcs_table.goodput_se(model_name, mcs_idx, snr_db)
        service_n = bandwidth_hat * se_eff * self.slot_duration / self.dpp_bit_scale
        arrivals_n = self.arrival_rate[user] * self.data_size[model_name] / self.dpp_bit_scale
        q_n = float(np.tanh(self.backlog[user] / self.dpp_bit_scale))
        z_n = float(np.tanh(self.energy_queue[user] / self.dpp_energy_scale))
        budget_n = self.energy_budget[user] / self.dpp_energy_scale
        energy_n = energy_hat / self.dpp_energy_scale
        return q_n * (service_n - arrivals_n) + z_n * (budget_n - energy_n)

    def forward_sim_mcs(self, user, snr_db, model_name, task_u):
        """ILLA MCS forward sim: BLER/SE + QoS only (no Acc-table scoring).

        Acc table is environment-only; learners observe Acc after realization.
        Vectorized over |MCS| when the per-slot overhead cache is active.
        """
        bler_t = getattr(self, 'bler_target', self.mcs_table.bler_target)
        if self._oh_user_base is not None:
            local_d, local_e, edge_d, edge_e, payload, bw, _backlog, p_tx = (
                self._oh_user_base[user][model_name])
            bler, goodput = self.mcs_table.bler_goodput_all_mcs(model_name, snr_db)
            mcs_idx = np.asarray(self.available_mcs, dtype=int)
            se = self.mcs_table._se_arr
            rates = bw * np.maximum(goodput, 1e-12)
            trans_d = payload / rates
            service = local_d + trans_d + edge_d
            energy = local_e + p_tx * trans_d + edge_e
            d_c = task_u['delay_constraint']
            e_c = task_u['energy_constraint']
            feas = (service <= d_c) & (energy <= e_c)
            if np.any(feas):
                under = feas & (bler <= bler_t)
                mask = under if np.any(under) else feas
                best_j = int(np.argmax(np.where(mask, se, -np.inf)))
                return int(mcs_idx[best_j])
            score = (task_u['delay_weight'] * erf(d_c - service)
                     + task_u['energy_weight'] * erf(e_c - energy))
            return int(mcs_idx[int(np.argmax(score))])

        feas = []
        best_infeas, best_infeas_score = None, -np.inf
        for mcs in self.available_mcs:
            service_hat, _, energy_hat = self.predict_md_overheads(
                user, None, model_name, mcs, snr_db=snr_db)
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

    def update_causal(self, snr_dic, mcs_dic, acc_dic):
        """Register realized interventional ACCURACY in the shared causal GP.

        The GP learns the mechanism-invariant P(acc | do(Model, MCS), SINR);
        per-MD QoS is composed analytically at arm-selection time so the scored
        objective matches get_reward without pooling heterogeneous rewards.

        Logs |acc - prior| / |acc - posterior| before optional register; ablations
        may skip ``add`` (no_update / freeze_after).

        Parallel update: pred-error batch solve charged /U; each MD's ``add``
        is independent work under a broadcast posterior → max over users.
        """
        records = [(user, mcs_dic[user], acc_dic[user]) for user in self.users]
        # Pred-error logging is diagnostic (not part of the online decision path).
        if self.log_pred_error:
            e_prior, e_post = self.causal_mab.prediction_errors(records)
            self.pred_err_prior.append(e_prior)
            self.pred_err_post.append(e_post)
        allow = self._mab_allow_update()
        add_times = []
        for user, mcs_realized, acc_obs in records:
            t_u = time.time()
            self.causal_mab.register_outcome(user, mcs_realized, acc_obs, do_update=allow)
            add_times.append(time.time() - t_u)
        # Window trim is ES-side (once per slot); amortize across MDs.
        t_trim = time.time()
        if allow:
            if self.causal_mab.shared:
                self.causal_mab.gp.maybe_trim()
            else:
                for gp in self.causal_mab._gps.values():
                    gp.maybe_trim()
        trim_s = (time.time() - t_trim) / max(len(records), 1)
        self._mab_update_slots += 1
        # Parallel MD view: each agent ships one obs; charge max local add cost.
        self._last_parallel_update_s = (
            (max(add_times) if add_times else 0.0) + trim_s)

    def realize_accuracy(self, acc_dic):
        """Add per-task observation noise to the curve-based accuracy values."""
        if self.acc_noise_std <= 0:
            return acc_dic
        return {
            user: float(np.clip(acc + np.random.normal(0.0, self.acc_noise_std), 0.0, 1.0))
            for user, acc in acc_dic.items()
        }


    def get_reward(self, task_dic, acc_dic, total_overhead_dic):
        """Utility: w_acc * accuracy + qos_coef * delay/energy erf terms."""
        w_acc = getattr(self, "reward_w_acc", 1.0)
        qos = getattr(self, "reward_qos_coef", 1.5)
        reward_dic = {}
        for user in self.users:
            reward_dic[user] = (
                w_acc * acc_dic[user]
                + qos * task_dic[user]["delay_weight"] * erf(
                    task_dic[user]["delay_constraint"] - total_overhead_dic[user]["delay"])
                + qos * task_dic[user]["energy_weight"] * erf(
                    task_dic[user]["energy_constraint"] - total_overhead_dic[user]["energy"]))
        return reward_dic

    def get_trans_rate(self, time_slot):
        """Read per-cell SINR from the trace and build Top-L candidate sets.

        Returns snr_dic, trans_rate_dic (best-candidate linear SINR for GP context),
        cand_cells_dic, sinr_db_all_dic.
        """
        trans_rate_dic = {}
        snr_dic = {}
        cand_cells_dic = {}
        sinr_db_all_dic = {}

        for user_idx, user in enumerate(self.users):
            top_cells = self.sinr_trace.top_cells(time_slot, user_idx, self.top_l_cells)
            sinr_vec = self.sinr_trace.sinr_vector(time_slot, user_idx)
            sinr_db_all = {}
            for cell_idx in range(self.sinr_trace.num_cells):
                snr_db = float(sinr_vec[cell_idx]) + self.sinr_offset_db
                if self.est_err_db > 0:
                    snr_db += self.est_err_db * np.random.randn()
                sinr_db_all[cell_idx] = snr_db

            cand_cells_dic[user] = top_cells
            sinr_db_all_dic[user] = sinr_db_all
            best_cell = top_cells[0]
            best_snr_linear = 10 ** (sinr_db_all[best_cell] / 10)
            snr_dic[user] = best_snr_linear
            trans_rate_dic[user] = best_snr_linear

        return snr_dic, trans_rate_dic, cand_cells_dic, sinr_db_all_dic

    def _goodput_se(self, user, model_name, mcs_idx, snr_dic):
        """Effective SE after TB erasures [bit/s/Hz]."""
        snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
        return max(self.mcs_table.goodput_se(model_name, mcs_idx, snr_db), 1e-12)

    def allocate_bandwidth(self, task_dic, model_selection_dic, trans_rate_dic, phy_choice_dic,
                           users=None, snr_dic=None):
        """Allocate bandwidth within one cell's pool (default: all users).

        Weights use goodput SE so erasures inflate the effective load."""
        users = self.users if users is None else users
        d_prime_dic = {}
        for user in users:
            chosen_model_m = model_selection_dic[user]["model"]
            p_m = self.md_params[user]['trans_power']
            omega_m_t = task_dic[user]['delay_weight']
            omega_m_e = task_dic[user]['energy_weight']
            q_n = self.backlog[user] / self.dpp_bit_scale
            se_eff = (self._goodput_se(user, chosen_model_m, phy_choice_dic[user], snr_dic)
                      if snr_dic is not None else max(self.mcs_table.se[phy_choice_dic[user]], 1e-12))
            d_prime_dic[user] = ((1 + q_n) * self.data_size[chosen_model_m]
                                 * (omega_m_t + p_m * omega_m_e) / se_eff)

        total_sqrt_d_prime = sum(np.sqrt(d) for d in d_prime_dic.values())
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
        """GPU frequency split among users associated to one cell."""
        users = self.users if users is None else users
        n = len(users)
        gpu_allocation_dict = {}

        f_m = cp.Variable(n)
        constraints = [0 <= f_m, f_m <= self.es_params['freq'], cp.sum(f_m) <= self.es_params['freq']]

        omega_m_t_vector = np.array([task_dic[user]['delay_weight'] for user in users])
        omega_m_e_vector = np.array([task_dic[user]['energy_weight'] for user in users])
        cores = self.es_params['cores']
        tail_flops_vector = np.array([self.tail_flops[model_selection_dic[user]["model"]] for user in users])
        flops_per_cycle = self.es_params['flops_per_cycle']
        power_coeff = self.es_params['power_coeff']

        first_term = omega_m_t_vector * tail_flops_vector @ cp.inv_pos(f_m) / (cores * flops_per_cycle)
        second_term = omega_m_e_vector * power_coeff @ (f_m ** 2) / (cores * flops_per_cycle)
        total_sum = cp.sum(first_term * 1e-9 + second_term)

        objective = cp.Minimize(total_sum)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS)
            if f_m.value is None:
                raise ValueError("SCS returned no solution")
        except Exception:
            problem.solve(solver=cp.ECOS)
        if f_m.value is None:
            for user in users:
                gpu_allocation_dict[user] = self.es_params['freq'] / n
        else:
            for idx, user in enumerate(users):
                gpu_allocation_dict[user] = f_m.value[idx]
        return gpu_allocation_dict

    def gpu_resource_allocation_all_cells(self, task_dic, model_selection_dic, cell_dic):
        """Per-cell GPU allocation; cells independent after association."""
        return bcd_loop.gpu_resource_allocation_all_cells(
            self, task_dic, model_selection_dic, cell_dic)

    def mcs_selection(self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                      local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic, users=None):
        """Select MCS: ILLA BLER filter + drift/QoS (no Acc-table term)."""
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

    def get_accuracy(self, snr_dic, phy_choice_dic, model_selection_dic):
        """Environment Acc realization from the PHY Acc table (not for decisions)."""
        acc_dic = {}
        for user in snr_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
            acc_dic[user] = self.mcs_table.accuracy(
                chosen_model_m, phy_choice_dic[user], snr_db)
        return acc_dic

    def get_local_overhead(self, model_selection_dic):
        """Calculate local processing overhead, including delay and energy consumption, for selected models."""

        local_overhead_dic = {}  # Dictionary to store delay and energy consumption for each user

        for user in self.users:
            chosen_model_m = model_selection_dic[user]["model"]  # Get the selected model for this user

            # Extract relevant parameters
            head_flops = self.head_flops[chosen_model_m]  # FLOPs of the head model
            flops_per_cycle = self.md_params[user]['flops_per_cycle']  # FLOPs per cycle for this user
            num_cores = self.md_params[user]['cores']  # Number of cores for this user
            gpu_freq = self.md_params[user]['freq']  # GPU frequency for this user

            # Compute local processing delay
            local_delay = head_flops * 1e-9 / (gpu_freq * num_cores * flops_per_cycle)

            # Compute local energy consumption
            local_energy = self.md_params[user]['power_coeff'] * gpu_freq ** 3 * local_delay

            # Store delay and energy in the result dictionary for the user
            local_overhead_dic[user] = {
                "delay": local_delay,
                "energy": local_energy
            }

        return local_overhead_dic  # Return dictionary with local overhead for all users

    def get_trans_overhead(self, trans_rate_dic, model_selection_dic, bandwidth_allocation_dic,
                           phy_choice_dic, snr_dic=None):
        """Transmission delay/energy using goodput SE (TB erasures)."""
        trans_overhead_dic = {}

        for user in trans_rate_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            bandwidth_m = bandwidth_allocation_dic[user]
            data_size_m = self.data_size[chosen_model_m]
            if snr_dic is not None:
                se_eff = self._goodput_se(user, chosen_model_m, phy_choice_dic[user], snr_dic)
            else:
                se_eff = max(self.mcs_table.se[phy_choice_dic[user]], 1e-12)
            # MCS: on-air payload delivery rate accounts for TB erasures via goodput
            trans_delay = data_size_m / (bandwidth_m * se_eff)
            trans_energy = self.md_params[user]['trans_power'] * trans_delay

            trans_overhead_dic[user] = {
                "delay": trans_delay,
                "energy": trans_energy
            }

        return trans_overhead_dic

    def get_edge_overhead(self, model_selection_dic, gpu_allocation_dic):
        """Calculate edge processing overhead, including delay and energy consumption, for selected models."""

        edge_overhead_dic = {}  # Dictionary to store delay and energy consumption for each user

        for user in self.users:
            chosen_model_m = model_selection_dic[user]["model"]  # Get the selected model for this user
            # Extract relevant parameters
            tail_flops_m = self.tail_flops[chosen_model_m]  # FLOPs of the tail model
            flops_per_cycle_m = self.es_params['flops_per_cycle']  # FLOPs per cycle for this user
            num_cores_m = self.es_params['cores']  # Number of cores for this user
            gpu_freq_m = gpu_allocation_dic[user]  # GPU frequency for this user

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
        total_overhead_dic = {}  # Dictionary to store delay and energy consumption for each user
        for user in self.users:
            queue_wait = 0.0 if queue_wait_dic is None else queue_wait_dic[user]
            total_delay = queue_wait + local_overhead_dic[user]['delay'] + trans_overhead_dic[user]['delay'] + \
                          edge_overhead_dic[user]['delay']
            total_energy = local_overhead_dic[user]['energy'] + trans_overhead_dic[user]['energy'] + \
                           edge_overhead_dic[user]['energy']
            # Store delay and energy in the result dictionary for the user
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
        """Calculate the average and standard deviation of latency, energy consumption, accuracy, reward,
        the probability of violating delay and energy consumption constraints, and
        the number of violations across all time slots and users."""

        metrics = ["delay", "energy", "accuracy", "reward", "is_vio", "vio_degree",
                   "backlog", "energy_queue", "arrivals", "served"]
        metric_sums = {m: 0 for m in metrics}
        metric_values = {m: [] for m in metrics}  # Store values for std calculation

        # Iterate over all users and time slots
        for user in self.users:
            for t in range(self.time_slot_num):
                for m in metrics:
                    value = self.instant_metrics[user][m][t]
                    metric_sums[m] += value
                    metric_values[m].append(value)

        total_samples = self.time_slot_num * len(self.users)

        # Compute averages
        self.average_metrics = {
            "latency": float(metric_sums["delay"] / total_samples),
            "energy": float(metric_sums["energy"] / total_samples),
            "accuracy": float(metric_sums["accuracy"] / total_samples),
            "reward": float(metric_sums["reward"] / total_samples),
            "vio_prob": float(metric_sums["is_vio"] / total_samples),
            "vio_sum": float(metric_sums["vio_degree"] / total_samples),
            "backlog_bits": float(metric_sums["backlog"] / total_samples),
            "energy_queue": float(metric_sums["energy_queue"] / total_samples),
            "arrival_bits": float(metric_sums["arrivals"] / total_samples),
            "served_bits": float(metric_sums["served"] / total_samples),
        }

        # Compute standard deviations
        self.std_metrics = {
            "latency": float(np.std(metric_values["delay"], ddof=1)),
            "energy": float(np.std(metric_values["energy"], ddof=1)),
            "accuracy": float(np.std(metric_values["accuracy"], ddof=1)),
            "reward": float(np.std(metric_values["reward"], ddof=1)),
            "vio_prob": float(np.std(metric_values["is_vio"], ddof=1)),
            "vio_sum": float(np.std(metric_values["vio_degree"], ddof=1)),
            "backlog_bits": float(np.std(metric_values["backlog"], ddof=1)),
            "energy_queue": float(np.std(metric_values["energy_queue"], ddof=1)),
        }

        # Normalize action frequency
        self.action_freq = self.action_freq / self.time_slot_num

    
    def get_instant_metrics(self,task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic=None):
        for user in self.users:
            self.instant_metrics[user]["delay"].append(total_overhead_dic[user]['delay'])
            self.instant_metrics[user]["energy"].append(total_overhead_dic[user]['energy'])
            self.instant_metrics[user]["accuracy"].append(acc_dic[user])
            self.instant_metrics[user]["reward"].append(reward_dic[user])  # Store reward for the current time slot
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
        """main loop for simulation"""
        for t in range(self.time_slot_num):
            snr_dic, trans_rate_dic, cand_cells_dic, sinr_db_all_dic = self.get_trans_rate(t)

            task_dic = self.generate_tasks(t)

            context_dic = self.observe_context(task_dic, trans_rate_dic)
            # distributed decision_ms = parallel (max-agent), not sequential sum.
            if self.algo == 'causal':
                model_selection_dic, cell_dic = self.model_selection_causal(
                    task_dic, cand_cells_dic, sinr_db_all_dic, trans_rate_dic)
            else:
                model_selection_dic, cell_dic = self.model_selection(
                    context_dic, task_dic, cand_cells_dic, sinr_db_all_dic)
            self.decision_time += float(self._last_parallel_decision_s)

            snr_dic = self._apply_cell_association(cell_dic, sinr_db_all_dic)

            arrival_bits_dic = {user: task_dic[user]["n_arrivals"]
                                * self.data_size[model_selection_dic[user]["model"]]
                                for user in self.users}

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

            # Env-only Acc realization (rewards / GP registration observe this)
            acc_dic = self.get_accuracy(snr_dic, phy_choice_dic, model_selection_dic)
            acc_realized_dic = self.realize_accuracy(acc_dic)

            # Calculate the performance for MDs
            reward_dic = self.get_reward(task_dic, acc_realized_dic, total_overhead_dic)

            # Queue dynamics: service drains the backlog, arrivals refill it;
            # service uses goodput SE so TB erasures reduce delivered bits.
            # the virtual energy queue tracks violations of the average budget
            service_bits_dic = {user: bandwidth_allocation_dic[user]
                                * self._goodput_se(user, model_selection_dic[user]["model"],
                                                   phy_choice_dic[user], snr_dic)
                                * self.slot_duration
                                for user in self.users}
            queue_info_dic = {}
            for user in self.users:
                self.backlog[user] = max(self.backlog[user] - service_bits_dic[user], 0.0) \
                    + arrival_bits_dic[user]
                self.energy_queue[user] = max(
                    self.energy_queue[user] + total_overhead_dic[user]['energy']
                    - self.energy_budget[user], 0.0)
                queue_info_dic[user] = {
                    "backlog": self.backlog[user], "energy_queue": self.energy_queue[user],
                    "arrivals": arrival_bits_dic[user], "served": service_bits_dic[user]}
            self.get_instant_metrics(task_dic, total_overhead_dic, reward_dic, acc_realized_dic,
                                     queue_info_dic)

            # Update the learning agents and cache the ES allocation for prediction
            # Parallel update_ms for factorized MDs (max-agent), folded into decision_ms.
            if self.algo == 'causal':
                self.update_causal(snr_dic, phy_choice_dic, acc_realized_dic)
            else:
                self.update_gp(context_dic, model_selection_dic, reward_dic)
            self._last_bandwidth = bandwidth_allocation_dic
            self._last_gpu = gpu_allocation_dic
            self.update_time += float(self._last_parallel_update_s)
        self.get_average_and_std_metrics()


if __name__ == "__main__":
    start_time = time.time()  # Record start time
    seed = 0
    config = Config(seed)
    omnis = OMNIS(config)
    omnis.simulation()
    print("aver info:", omnis.average_metrics)
    print("std info:", omnis.std_metrics)
    print("action freq info:", omnis.action_freq)
    end_time = time.time()  # Record end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    # omnis.show_convergence()

