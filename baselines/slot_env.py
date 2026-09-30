import numpy as np
from scipy.special import erf
import time
import matplotlib.pyplot as plt
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


class SlotEnv:
    """Shared multi-cell slot loop for baselines that do not use OMNIS.

    Branch and cell decisions are supplied by the subclass. This class is
    the environment, not a comparison scheme.
    """

    def __init__(self, config):
        """Initialize system parameters including models, devices, and users."""
        # Basic configuration
        self.name = "slot"
        self.seed = config.seed
        np.random.seed(self.seed)  # Set random seed for reproducibility

        # Computation-related parameters
        self.models = config.models
        self.data_size = config.data_size
        self.head_flops = config.head_flops
        self.tail_flops = config.tail_flops


        # User and system parameters
        self.users = config.users
        self.user_num = config.user_num
        self.md_params = config.md_params
        self.es_params = config.es_params
        self.time_slot_num = config.time_slot_num

        # Network and communication parameters
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
        self.action_freq = config.action_freq
        self.decision_time = 0.0
        self.update_time = 0.0
        self.bcd_time = 0.0
        self.bcd_iters = 0.0  # running avg BCD iterations / slot
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

        # BCD (Block Coordinate Descent) algorithm settings
        self.bcd_flag = config.bcd_flag
        self.bcd_max_iter = config.bcd_max_iter

        # Queueing model + Lyapunov framework (journal extension)
        self.slot_duration = getattr(config, 'slot_duration', 1.0)
        self.arrival_rate = getattr(config, 'arrival_rate', {user: 0.7 for user in self.users})
        self.energy_budget = getattr(config, 'energy_budget', {user: 0.45 for user in self.users})
        self.lyapunov_v = getattr(config, 'lyapunov_v', 1.0)
        self.reward_w_acc = getattr(config, 'reward_w_acc', 1.0)
        self.reward_qos_coef = getattr(config, 'reward_qos_coef', 1.5)
        self.dpp_bit_scale = getattr(config, 'dpp_bit_scale', 2.2e4)
        self.dpp_energy_scale = getattr(config, 'dpp_energy_scale', 0.45)
        init_task_pipeline(self, config)
        self._last_bandwidth = {}
        self._last_gpu = {}
        self._last_mcs = {}
        self._cell_compute_state = init_cell_compute_state(
            self.num_cells, self.es_params['freq'], self.user_num)
        self.log_proxy_calib = bool(getattr(config, 'log_proxy_calib', False))
        self.proxy_calib = []
        self.compute_realism_logstd = float(
            getattr(config, 'compute_realism_logstd', 0.0))

    def _payload_bits(self, model_name):
        return payload_bits(self.data_size, model_name)

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

    def model_selection(self, cand_cells_dic):
        """Subclass hook: return (model_selection_dic, cell_dic)."""
        del cand_cells_dic
        raise NotImplementedError(
            "SlotEnv has no branch policy; use a baseline subclass")

    def _gpu_hat_for_association(self, user, cell_id=None):
        if cell_id is not None:
            return expected_gpu_if_join(
                cell_id, self._cell_compute_state,
                self.es_params['freq'], self.user_num)
        return float(self._last_gpu.get(
            user, self.es_params['freq'] / max(self.user_num, 1)))

    def predict_md_overheads(self, user, rate_m, model_name, mcs_idx, snr_db=0.0,
                             cell_id=None):
        """Same measurement-driven hats as OMNIS+ (fair comparison)."""
        del rate_m
        from omnis.compute_stats import predict_service_overheads
        return predict_service_overheads(
            self, user, model_name, mcs_idx, snr_db=snr_db, cell_id=cell_id)


    def dpp_drift(self, user, model_name, mcs_idx, energy_hat, snr_db=0.0, cell_id=None):
        return task_dpp_drift(
            self, user, model_name, mcs_idx, energy_hat, snr_db=snr_db, cell_id=cell_id)

    def forward_sim_mcs(self, user, snr_db, model_name, task_u, cell_id=None):
        """ILLA MCS forward sim: BLER/SE + QoS only (no Acc-table scoring).

        Acc table is environment-only; learners observe Acc after realization.
        """
        bler_t = getattr(self, 'bler_target', self.mcs_table.bler_target)
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
        """Continuous GPU frequency share among concurrent edge-stage MDs."""
        from omnis.gpu_alloc import allocate_edge_gpu
        from omnis.compute_stats import ensure_compute_stats
        users = self.users if users is None else users
        return allocate_edge_gpu(
            getattr(self, "pipeline", None),
            self.es_params,
            task_dic,
            model_selection_dic,
            users,
            self.num_cells,
            ensure_compute_stats(self),
        )


    def gpu_resource_allocation_all_cells(self, task_dic, model_selection_dic, cell_dic):
        """Per-cell GPU allocation; cells independent after association."""
        return bcd_loop.gpu_resource_allocation_all_cells(
            self, task_dic, model_selection_dic, cell_dic)

    def mcs_selection(self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                      local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic, users=None):
        """ILLA BLER filter + drift/QoS score (no Acc-table term)."""
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
                cell_id = None
                pipe = getattr(self, "pipeline", None)
                if pipe is not None:
                    locked = pipe.locked_cell(user)
                    if locked is not None:
                        cell_id = int(locked)
                drift = self.dpp_drift(
                    user, model_name_u, mcs, total_energy,
                    snr_db=snr_db, cell_id=cell_id)
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
        """Local residual delay/energy from online stats / observed residual."""
        from omnis.task_pipeline import STAGE_LOCAL
        from omnis.compute_stats import ensure_compute_stats

        stats = ensure_compute_stats(self)
        local_overhead_dic = {}
        for user in model_selection_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            full_d, full_e = stats.local_hat(user, chosen_model_m)
            at = self.pipeline.active.get(user)
            if at is None:
                local_delay, local_energy = full_d, full_e
            elif at.stage == STAGE_LOCAL:
                local_delay = float(at.residual)
                local_energy = full_e * min(local_delay / max(full_d, 1e-12), 1.0)
            else:
                local_delay, local_energy = 0.0, 0.0
            local_overhead_dic[user] = {"delay": local_delay, "energy": local_energy}
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
            from omnis.compute_stats import tx_power_forecast
            pipe = getattr(self, "pipeline", None)
            cell = pipe.locked_cell(user) if pipe is not None else None
            p_tx = tx_power_forecast(
                self, user, cell_id=cell, bw_hz=bandwidth_m)
            trans_energy = p_tx * trans_delay
            trans_overhead_dic[user] = {"delay": trans_delay, "energy": trans_energy}
        return trans_overhead_dic


    def get_edge_overhead(self, model_selection_dic, gpu_allocation_dic):
        """Edge delay/energy from online work stats (no FLOPs maps)."""
        from omnis.compute_stats import ensure_compute_stats

        stats = ensure_compute_stats(self)
        edge_overhead_dic = {}
        for user in model_selection_dic.keys():
            chosen_model_m = model_selection_dic[user]["model"]
            gpu_freq_m = max(float(gpu_allocation_dic.get(user, 0.0)), 0.0)
            if gpu_freq_m <= 1e-12:
                from omnis.gpu_alloc import forecast_gpu_share
                from omnis.task_pipeline import STAGE_EDGE
                pipe = getattr(self, "pipeline", None)
                cell = pipe.locked_cell(user) if pipe is not None else None
                q = pipe.compute_queue_len(cell) if cell is not None else 0
                at = pipe.active.get(user) if pipe is not None else None
                in_edge = at is not None and at.stage == STAGE_EDGE
                gpu_freq_m = forecast_gpu_share(
                    q, float(self.es_params['freq']), already_in_edge=in_edge)
            edge_delay, edge_energy = stats.edge_hat(chosen_model_m, gpu_freq_m)
            edge_overhead_dic[user] = {"delay": edge_delay, "energy": edge_energy}
        return edge_overhead_dic

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
