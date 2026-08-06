import numpy as np
import time
from scipy.special import erf
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from sys_data.config import Config
from omnis import cbo
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
seed = 42
np.random.seed(seed)


class DppUtility:
    """Wraps a base acquisition utility with Lyapunov V scaling and joint drift."""

    def __init__(self, base, V, drift_fn):
        self.base, self.V, self.drift_fn = base, V, drift_fn

    def utility(self, x, gp):
        return self.V * self.base.utility(x, gp) + self.drift_fn(x)


class CTO:
    def __init__(self, config):
        """Initialize system parameters including models, devices, and users."""
        # Basic information
        self.name = "cto"
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

        # Performance metrics
        self.instant_metrics = config.instant_metrics
        self.average_metrics = config.average_metrics
        self.std_metrics = config.std_metrics
        self.est_err = config.est_err
        self.est_err_db = getattr(config, 'est_err_db', 1.0)
        self.sinr_trace = config.sinr_trace
        self.top_l_cells = config.top_l_cells
        self.num_cells = config.num_cells
        # Std of the per-task accuracy observation noise (0 disables)
        self.acc_noise_std = getattr(config, 'acc_noise_std', 0.0)
        self.action_freq = config.action_freq
        self.decision_time = 0.0
        self.bcd_time = 0.0
        self.update_time = 0.0

        # BCD (Block Coordinate Descent) algorithm parameters
        self.bcd_flag = config.bcd_flag
        self.bcd_max_iter = config.bcd_max_iter

        self.action = {}
        for user in self.users:
            self.action[f"{user}_model"] = np.arange(len(self.models))
            self.action[f"{user}_cell_rank"] = np.arange(self.top_l_cells)

        self.contexts = {
            key: ''
            for user in self.users
            for key in [
                f"{user}_delay_constraint",
                f"{user}_energy_constraint",
                f"{user}_transmission_rate",
                f"{user}_energy_weight",
                f"{user}_delay_weight"
            ]
        }

        self.action_dim = len(self.action)
        self.context_dim = len(self.contexts)

        # GP kernel settings
        self.length_scale = np.ones(self.context_dim + self.action_dim)
        self.kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=self.length_scale)
        self.noise = config.noise
        self.beta_function = config.beta_function
        self.beta_const_val = config.beta_const_val
        # Joint space is (n_models * L)^U — never materialize; sample K on the fly.
        # GP hypers: cto_gp_burn_in<=0 keeps full L-BFGS ARD every slot (joint CBO
        # cost); positive N freezes after burn-in. Stock sklearn predict (not FastGP)
        # so scoring K joint candidates retains centralized wall time. No sleep().
        self.max_candidates = getattr(config, 'cto_max_candidates', None)
        self.gp_burn_in = int(getattr(config, 'cto_gp_burn_in', 0))
        self.gp_n_restarts = int(getattr(config, 'cto_gp_n_restarts', 5))
        self.use_fast_gp = bool(getattr(config, 'cto_use_fast_gp', False))
        self.optimizer = cbo.ContextualBayesianOptimization(
            all_actions_dict=self.action, contexts=self.contexts, kernel=self.kernel,
            gp_burn_in=self.gp_burn_in, n_restarts_optimizer=self.gp_n_restarts,
            use_fast_gp=self.use_fast_gp)
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
        self.backlog = {user: 0.0 for user in self.users}
        self.energy_queue = {user: 0.0 for user in self.users}
        self._last_bandwidth = {}
        self._last_gpu = {}

    def generate_tasks(self, time_slot):
            """Dynamically adjust delay and energy constraints while keeping the base values fixed.
            Also draws the Poisson task arrivals for this slot (queueing model)."""

            task_dic = {
                user: {
                    "delay_constraint": self.fixed_delay[user]+ np.random.uniform(-0.001, 0.001),
                    "energy_constraint": self.fixed_energy[user]+ np.random.uniform(-0.001, 0.001),
                    "energy_weight": self.fixed_energy_weight[user]+ np.random.uniform(-0.001, 0.001),
                    "n_arrivals": np.random.poisson(self.arrival_rate[user]),
                }
                for user in self.users
            }

            for user in self.users:
                task_dic[user]["delay_weight"] = 1 - task_dic[user]["energy_weight"]

            return task_dic

    def observe_context(self, task_dic, trans_rate_dic):
        """Observe the current context for all users and store them in a dictionary with unique keys for each user."""

        # Create a dictionary to store the context for each user
        context_dict = {}

        for user in self.users:
            # Extract the context for each user and assign to the appropriate keys
            context_dict[f"{user}_delay_constraint"] = task_dic[user]["delay_constraint"]
            context_dict[f"{user}_energy_constraint"] = task_dic[user]["energy_constraint"]
            context_dict[f"{user}_transmission_rate"] = trans_rate_dic[user]
            context_dict[f"{user}_energy_weight"] = task_dic[user]["energy_weight"]
            context_dict[f"{user}_delay_weight"] = task_dic[user]["delay_weight"]

        return context_dict

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
        """Joint (model, cell_rank) selection via CBO with DPP drift in acquisition."""
        action_keys = list(self.optimizer._space._action_keys)
        model_key_idx = {user: action_keys.index(f'{user}_model') for user in self.users}
        cell_key_idx = {user: action_keys.index(f'{user}_cell_rank') for user in self.users}

        # Drift is additive over users and independent of other users' arms.
        # Precompute the U × n_models × L table once per slot, then gather —
        # math-equivalent to the per-candidate nested loop, but O(U·M·L) MCS /
        # overhead calls instead of O(K·U).
        n_models = len(self.models)
        L = self.top_l_cells
        n_users = len(self.users)
        drift_table = np.zeros((n_users, n_models, L), dtype=np.float64)
        for ui, user in enumerate(self.users):
            task_u = task_dic[user]
            for model_idx in range(n_models):
                model_name = self.models[model_idx]["name"]
                for cell_rank in range(L):
                    cell_id = cand_cells_dic[user][cell_rank]
                    snr_db = sinr_db_all_dic[user][cell_id]
                    mcs_hat = self.forward_sim_mcs(user, snr_db, model_name, task_u)
                    _, _, energy_hat = self.predict_md_overheads(
                        user, None, model_name, mcs_hat, snr_db=snr_db)
                    drift_table[ui, model_idx, cell_rank] = self.dpp_drift(
                        user, model_name, mcs_hat, energy_hat, snr_db=snr_db)

        user_model_cols = np.array(
            [model_key_idx[u] for u in self.users], dtype=np.int64)
        user_cell_cols = np.array(
            [cell_key_idx[u] for u in self.users], dtype=np.int64)

        def joint_drift(context_action):
            action_cols = context_action[:, self.context_dim:]
            drifts = np.zeros(len(context_action), dtype=np.float64)
            for ui in range(n_users):
                m = np.asarray(action_cols[:, user_model_cols[ui]], dtype=np.int64)
                c = np.asarray(action_cols[:, user_cell_cols[ui]], dtype=np.int64)
                drifts += drift_table[ui, m, c]
            return drifts

        dpp_utility = DppUtility(self.utility, self.lyapunov_v, joint_drift)
        action_dic = self.optimizer.suggest(context_dic, dpp_utility,
                                            max_candidates=self.max_candidates)

        model_selection_dic = {}
        cell_dic = {}
        for user_idx, user in enumerate(self.users):
            selected_model_m = action_dic[f'{user}_model']
            selected_cell_rank = action_dic[f'{user}_cell_rank']
            cell_id = cand_cells_dic[user][selected_cell_rank]
            model_selection_dic[user] = {
                "model": self.models[selected_model_m]["name"],
                "cell_rank": selected_cell_rank,
            }
            cell_dic[user] = cell_id
            self.action_freq[user_idx, selected_model_m, selected_cell_rank] += 1
        return action_dic, model_selection_dic, cell_dic

    def update_gp(self, context_dic, action_dic, reward_dic):
        """Update the GP model with new observations."""
        average_reward = np.mean(list(reward_dic.values()))
        self.optimizer.register(context_dic, action_dic, average_reward)

    def predict_md_overheads(self, user, rate_m, model_name, mcs_idx, snr_db=0.0):
        """Analytic Payload -> {Delay, Energy}; transmission uses goodput SE."""
        md = self.md_params[user]

        head_flops = self.head_flops[model_name]
        local_delay = head_flops * 1e-9 / (md['freq'] * md['cores'] * md['flops_per_cycle'])
        local_energy = md['power_coeff'] * md['freq'] ** 3 * local_delay

        bandwidth_hat = self._last_bandwidth.get(user, self.total_bandwidth / self.user_num)
        se_eff = self.mcs_table.goodput_se(model_name, mcs_idx, snr_db)
        rate_hat = bandwidth_hat * max(se_eff, 1e-12)
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
        """Analytic Lyapunov drift; service uses goodput SE."""
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
        """ILLA-style MCS forward sim with QoS feasibility."""
        bler_t = getattr(self, 'bler_target', self.mcs_table.bler_target)
        feas = []
        best_infeas, best_infeas_score = None, -np.inf
        for mcs in self.available_mcs:
            service_hat, _, energy_hat = self.predict_md_overheads(
                user, None, model_name, mcs, snr_db=snr_db)
            acc_hat = self.mcs_table.accuracy(model_name, mcs, snr_db)
            if (service_hat <= task_u['delay_constraint']
                    and energy_hat <= task_u['energy_constraint']):
                bler = self.mcs_table.bler(model_name, mcs, snr_db)
                feas.append((mcs, bler, self.mcs_table.se[mcs], acc_hat))
            else:
                score = (acc_hat
                         + task_u['delay_weight'] * erf(task_u['delay_constraint'] - service_hat)
                         + task_u['energy_weight'] * erf(task_u['energy_constraint'] - energy_hat))
                if score > best_infeas_score:
                    best_infeas, best_infeas_score = mcs, score
        if feas:
            under = [t for t in feas if t[1] <= bler_t]
            pool = under if under else feas
            return max(pool, key=lambda t: (t[2], t[3]))[0]
        return best_infeas


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
        """Read per-cell SINR from the trace and build Top-L candidate sets."""
        trans_rate_dic = {}
        snr_dic = {}
        cand_cells_dic = {}
        sinr_db_all_dic = {}

        for user_idx, user in enumerate(self.users):
            top_cells = self.sinr_trace.top_cells(time_slot, user_idx, self.top_l_cells)
            sinr_vec = self.sinr_trace.sinr_vector(time_slot, user_idx)
            sinr_db_all = {}
            for cell_idx in range(self.sinr_trace.num_cells):
                snr_db = float(sinr_vec[cell_idx])
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
        """Allocate bandwidth within one cell's pool (goodput-weighted)."""
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
        """Per-cell bandwidth allocation; each cell has a full bandwidth pool."""
        bandwidth_allocation_dic = {}
        for _cell, users in self._users_by_cell(cell_dic).items():
            bandwidth_allocation_dic.update(
                self.allocate_bandwidth(task_dic, model_selection_dic, trans_rate_dic,
                                        phy_choice_dic, users=users, snr_dic=snr_dic))
        return bandwidth_allocation_dic


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
        """Per-cell GPU allocation; each cell has a full ES GPU pool."""
        gpu_allocation_dic = {}
        for _cell, users in self._users_by_cell(cell_dic).items():
            gpu_allocation_dic.update(
                self.gpu_resource_allocation(task_dic, model_selection_dic, users=users))
        return gpu_allocation_dic

    def mcs_selection(self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                      local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic, users=None):
        """ILLA BLER filter + DPP score among QoS-feasible MCS."""
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
                acc_dic = self.get_accuracy(
                    {user: snr_dic[user]}, temp_mcs_dic,
                    {user: model_selection_dic[user]})
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
                    score = self.lyapunov_v * getattr(self, "reward_w_acc", 1.0) * acc_dic[user] + drift
                    bler = self.mcs_table.bler(model_name_u, mcs, snr_db)
                    entry = (mcs, score, bler, self.mcs_table.se[mcs])
                    if bler <= bler_t:
                        under.append(entry)
                    else:
                        over.append(entry)
                else:
                    score = (self.lyapunov_v * (getattr(self, "reward_w_acc", 1.0) * acc_dic[user]
                             + task_dic[user]['delay_weight']
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
        """Per-cell MCS selection using each user's associated-cell SINR."""
        mcs_dic = {}
        for _cell, users in self._users_by_cell(cell_dic).items():
            mcs_dic.update(self.mcs_selection(
                task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic,
                users=users))
        return mcs_dic

    def get_accuracy(self, snr_dic, mcs_dic, model_selection_dic):
        """ Get the table accuracy for a given model, MCS, and SNR."""
        acc_dic = {}
        for user in snr_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            snr_db = 10 * np.log10(max(snr_dic[user], 1e-12))
            acc_dic[user] = self.mcs_table.accuracy(chosen_model_m, mcs_dic[user], snr_db)
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
                           mcs_dic, snr_dic=None):
        """Transmission delay/energy using goodput SE (TB erasures)."""
        trans_overhead_dic = {}
        for user in trans_rate_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            bandwidth_m = bandwidth_allocation_dic[user]
            data_size_m = self.data_size[chosen_model_m]
            if snr_dic is not None:
                se_eff = self._goodput_se(user, chosen_model_m, mcs_dic[user], snr_dic)
            else:
                se_eff = max(self.mcs_table.se[mcs_dic[user]], 1e-12)
            trans_delay = data_size_m / (bandwidth_m * se_eff)
            trans_energy = self.md_params[user]['trans_power'] * trans_delay
            trans_overhead_dic[user] = {"delay": trans_delay, "energy": trans_energy}
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

            # Compute local processing delay
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
        """main loop for simulation"""

        for t in range(self.time_slot_num):
            snr_dic, trans_rate_dic, cand_cells_dic, sinr_db_all_dic = self.get_trans_rate(t)

            task_dic = self.generate_tasks(t)

            context_dic = self.observe_context(task_dic, trans_rate_dic)
            t_decision = time.time()
            action_dic, model_selection_dic, cell_dic = self.model_selection(
                context_dic, task_dic, cand_cells_dic, sinr_db_all_dic)
            self.decision_time += time.time() - t_decision
            snr_dic = self._apply_cell_association(cell_dic, sinr_db_all_dic)

            arrival_bits_dic = {user: task_dic[user]["n_arrivals"]
                                * self.data_size[model_selection_dic[user]["model"]]
                                for user in self.users}

            # Calculate the local processing overhead
            local_overhead_dic = self.get_local_overhead(model_selection_dic)

            # The ES performs BCD-based optimization
            t_bcd = time.time()
            bcd_obj_last = float('inf')  # Previous objective function value (used for convergence check)
            bcd_iter = 1  # Iteration counter

            while True:
                # Choose initialization or update step based on the current iteration
                if bcd_iter == 1:
                    init_mcs_dic = {user: random.choice(self.available_mcs) for user in self.users}
                    bandwidth_allocation_dic = self.allocate_bandwidth_all_cells(
                        task_dic, model_selection_dic, trans_rate_dic, init_mcs_dic, cell_dic,
                        snr_dic=snr_dic)
                else:
                    bandwidth_allocation_dic = self.allocate_bandwidth_all_cells(
                        task_dic, model_selection_dic, trans_rate_dic, phy_choice_dic, cell_dic,
                        snr_dic=snr_dic)

                gpu_allocation_dic = self.gpu_resource_allocation_all_cells(
                    task_dic, model_selection_dic, cell_dic)

                phy_choice_dic = self.mcs_selection_all_cells(
                    task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                    local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic, cell_dic)

                # Get the accuracy for each user based on current SNR, MCS, and model selection
                acc_dic = self.get_accuracy(snr_dic, phy_choice_dic, model_selection_dic)

                # Get the transmission overhead for each user
                trans_overhead_dic = self.get_trans_overhead(trans_rate_dic, model_selection_dic,
                                                             bandwidth_allocation_dic, phy_choice_dic,
                                                             snr_dic=snr_dic)

                # Get the edge processing overhead for each user
                edge_overhead_dic = self.get_edge_overhead(model_selection_dic, gpu_allocation_dic)

                # Calculate the total overhead for each user (queue wait + local + transmission + edge)
                queue_wait_dic = {user: self.backlog[user] / (bandwidth_allocation_dic[user] * self._goodput_se(user, model_selection_dic[user]["model"], phy_choice_dic[user], snr_dic))
                                  for user in self.users}
                total_overhead_dic = self.get_total_overhead(local_overhead_dic, trans_overhead_dic,
                                                             edge_overhead_dic, queue_wait_dic)
                # Find the user with the minimum accuracy and the corresponding accuracy value
                bcd_min_acc_user = min(acc_dic, key=lambda user: float(acc_dic[user]))
                bcd_min_acc_value = float(acc_dic[bcd_min_acc_user])

                # Calculate the delay penalty for each user (how much it exceeds the delay constraint)
                bcd_delay_penalty = sum(
                    erf(total_overhead_dic[user]['delay'] - task_dic[user]['delay_constraint']) for user in
                    self.users)

                # Calculate the energy penalty for each user (how much it exceeds the energy constraint)
                bcd_energy_penalty = sum(
                    erf(total_overhead_dic[user]['energy'] - task_dic[user]['energy_constraint']) for user in
                    self.users)

                # Calculate the total objective function value
                bcd_obj = bcd_min_acc_value + bcd_delay_penalty + bcd_energy_penalty
                # print("BCD obj:",bcd_obj)
                # Check convergence condition (if objective function change is small or max iterations reached)
                if abs(bcd_obj - bcd_obj_last) <= self.bcd_flag or bcd_iter >= self.bcd_max_iter:
                    break

                # Update iteration counter and last objective function value for the next iteration
                bcd_iter += 1
                bcd_obj_last = bcd_obj
            self.bcd_time += time.time() - t_bcd

            # Record the realized PHY diagnostics
            for user in self.users:
                self.instant_metrics[user]["mcs"].append(phy_choice_dic[user])
                self.instant_metrics[user]["cell"].append(cell_dic[user])
                snr_db_u = 10 * np.log10(max(snr_dic[user], 1e-12))
                self.instant_metrics[user]["bler"].append(self.mcs_table.bler(
                    model_selection_dic[user]["model"], phy_choice_dic[user], snr_db_u))

            # Realize the per-task accuracy (curve mean + observation noise)
            if self.acc_noise_std > 0:
                acc_dic = {
                    user: float(np.clip(acc + np.random.normal(0.0, self.acc_noise_std), 0.0, 1.0))
                    for user, acc in acc_dic.items()
                }

            # Calculate the reward for MDs and update the GP
            reward_dic = self.get_reward(task_dic, acc_dic, total_overhead_dic)

            service_bits_dic = {user: bandwidth_allocation_dic[user] * self._goodput_se(user, model_selection_dic[user]["model"], phy_choice_dic[user], snr_dic) * self.slot_duration
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
            self.get_instant_metrics(task_dic, total_overhead_dic, reward_dic, acc_dic, queue_info_dic)
            self._last_bandwidth = bandwidth_allocation_dic
            self._last_gpu = gpu_allocation_dic
            t_update = time.time()
            self.update_gp(context_dic, action_dic, reward_dic)
            self.update_time += time.time() - t_update

        self.get_average_and_std_metrics()

if __name__ == "__main__":
    start_time = time.time()  # Record start time
    seed = 42
    config = Config(seed)
    cto = CTO(config)
    cto.simulation()
    print("aver info:", cto.average_metrics)
    print("std info:", cto.std_metrics)
    print("action freq info:", cto.action_freq)

    end_time = time.time()  # Record end time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    cto.show_convergence()
