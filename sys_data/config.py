import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
import omnis.cbo as cbo
import omnis.util as util
from omnis.mcs_table import McsTable


class Config:
    def __init__(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.time_slot_num = 500

        self.models = [
            {'name': f'{q}{c}', 'quant_method': q, 'quant_channel': c}
            for q, c in [('Box', 3), ('Box', 6), ('Box', 12),
                         ('Standard', 3), ('Standard', 6), ('Standard', 12)]
        ]

        # On-air payload [bytes]; queues use bits = 8 * data_size.
        self.data_size = {
            'Box3': 6e3, 'Box6': 13.26e3, 'Box12': 33.58e3,
            'Standard3': 11.23e3, 'Standard6': 22.46e3, 'Standard12': 44.93e3
        }
        self.data_size_bits = {k: 8.0 * v for k, v in self.data_size.items()}

        # Mask R-CNN / ResNet50 split FLOPs (shared across branches).
        _head, _tail = 2.84e9, 3.71e11
        self.head_flops = {m['name']: _head for m in self.models}
        self.tail_flops = {m['name']: _tail for m in self.models}

        self.ue_pool_size = 25
        self.user_num = self.ue_pool_size
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]

        _md = {
            'freq': 1.6, 'cores': 768, 'flops_per_cycle': 12,
            'power_coeff': 0.35, 'trans_power': 0.1,
        }
        self.md_params = {user: dict(_md) for user in self.users}
        self.es_params = {
            'freq': 4.0, 'cores': 12288, 'flops_per_cycle': 24,
            'power_coeff': 0.75,
        }
        self.md_params_origin = self.md_params
        self.es_params_origin = self.es_params

        self.delay_constraint_range = (1.60, 2.50)
        self.energy_constraint_range = (0.11, 0.26)
        self.energy_weight_range = (0.25, 0.45)
        self.fixed_delay = {
            user: np.random.uniform(*self.delay_constraint_range)
            for user in self.users}
        self.fixed_energy = {
            user: np.random.uniform(*self.energy_constraint_range)
            for user in self.users}
        self.fixed_energy_weight = {
            user: np.random.uniform(*self.energy_weight_range)
            for user in self.users}

        self.slot_duration = 1.0
        self.arrival_rate = {user: 0.07 for user in self.users}
        self.energy_budget = {user: 3.0 for user in self.users}
        self.arrival_rate_origin = self.arrival_rate
        self.energy_budget_origin = self.energy_budget
        self.lyapunov_v = 10.0
        self.reward_w_acc = 24.0
        self.reward_qos_coef = 20.0
        self.causal_drift_gain = 1.5
        self.mab_no_update = False
        self.mab_freeze_after = 0
        self.log_pred_error = True
        self.dpp_bit_scale = float(np.mean(list(self.data_size_bits.values())))
        self.dpp_task_scale = 3.0
        self.dpp_energy_scale = 1.0
        self.learn_delay_slots = 1

        self.fixed_delay_origin = self.fixed_delay
        self.fixed_energy_origin = self.fixed_energy
        self.fixed_energy_weight_origin = self.fixed_energy_weight

        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [],
                   "vio_degree": [], "reward": [], "mcs": [], "bler": [],
                   "backlog": [], "energy_queue": [], "arrivals": [],
                   "served": [], "cell": []}
            for user in self.users
        }

        self.total_bandwidth = 6.5e5
        self.est_err = 0.5
        self.est_err_db = 1.0
        self.sinr_offset_db = 0.0
        self.noise_power_dBm = -174 + 10 * np.log10(self.total_bandwidth)
        self.noise_power = 10 ** (self.noise_power_dBm / 10 - 3)

        self.num_cells = 7
        self.top_l_cells = 3
        self.assoc_w_radio = 1.0
        self.assoc_w_compute = 1.0
        self.sinr_trace_tag = "smoke7"
        self.sinr_trace_dir = "phy_sim/output"
        self.phy_mode = "su_mimo"
        self.phy_bs_rows = 2
        self.phy_bs_cols = 4
        self.phy_ue_ants = 2
        self.phy_n_layers = 2
        from omnis.sinr_trace import SinrTrace, select_spread_ue_ids
        _probe = SinrTrace.from_config_dir(
            self.sinr_trace_dir, tag=self.sinr_trace_tag)
        self.sinr_ue_pool = select_spread_ue_ids(
            self.ue_pool_size, _probe.num_ues, _probe.num_cells)
        self.sinr_trace = SinrTrace.from_config_dir(
            self.sinr_trace_dir, tag=self.sinr_trace_tag,
            ue_ids=self.sinr_ue_pool[:self.user_num])
        if getattr(self.sinr_trace, "meta", None):
            self.phy_mode = self.sinr_trace.meta.get("phy", self.phy_mode)
            self.phy_ue_ants = int(self.sinr_trace.meta.get(
                "ue_ants", self.phy_ue_ants))
            self.phy_n_layers = int(self.sinr_trace.meta.get(
                "n_layers", self.phy_n_layers))

        self.mcs_table = McsTable("phy_sim/output", acc_floor=0.0, bler_target=0.1)
        if getattr(self.sinr_trace, "meta", None):
            self.mcs_table.n_streams = int(
                self.sinr_trace.meta.get("rate_layers", 1))
        self.available_mcs = list(range(self.mcs_table.num_mcs))
        self.bler_target = self.mcs_table.bler_target

        self.bcd_flag = 10e-5
        self.bcd_max_iter = 30
        self.average_metrics = {}
        self.std_metrics = {}
        self.rewards_history = {user: [] for user in self.users}

        self._init_action_and_gp()
        self.utility = util.UtilityFunction(
            kind="ucb", beta_kind=self.beta_function,
            beta_const=self.beta_const_val)

        self.algo = 'causal'
        self.acc_noise_std = 0.02
        self.causal_acq = 'ucb'
        self.causal_use_prior = False
        self.causal_shared = True
        self.causal_explore_slots = 80
        self.causal_init_random = 2
        self.causal_empty_prior_std = 1.0
        self.causal_gp_max_obs = 500
        self.causal_beta = 0.12
        self.causal_feas_margin = 0.88
        self.causal_gp_length_scales = [6.0, 0.40, 2.0, 0.20]
        self.causal_gp_signal_var = 4.0e-2
        self.cto_max_candidates = 12288
        self.cto_gp_n_restarts = 2
        self.cto_use_fast_gp = True
        self.gdo_resource_resolution = 8
        self.gdo_price_radio = 1.0
        self.gdo_price_compute = 1.0
        self.gdo_offer = 2.0
        self.gdo_mab_beta = 0.25
        self.gp_init_random = 25
        self.comm_rtt_s = 5e-4
        self.comm_ctrl_rate_bps = 2e6
        self.rl_dpp_reward = True
        self.rl_eval = False
        self.rl_hidden = 64
        self.rl_gamma = 0.99
        self.rl_reward_norm = True
        self.rl_log_every = 50
        self.dqn_pretrain_slots = 400
        self.dqn_model_path = "figures/dqn_pretrained.pt"
        self.dqn_eval = False
        self.dqn_dpp_reward = True
        self.dqn_gamma = 0.99
        self.dqn_batch_size = 64
        self.dqn_buffer_size = 6000
        self.dqn_lr = 5e-4
        self.dqn_target_sync = 40
        self.dqn_eps_start = 1.0
        self.dqn_eps_end = 0.05
        self.dqn_eps_decay_slots = 300
        self.dqn_train_every = 2
        self.dqn_grad_steps = 1
        self.dqn_hidden = 64
        self.dqn_max_candidates = 2000
        self.dqn_train_candidates = 256
        self.ppo_eval = False
        self.ppo_dpp_reward = True
        self.ppo_gamma = 0.99
        self.ppo_gae_lambda = 0.95
        self.ppo_clip_eps = 0.2
        self.ppo_entropy_coef = 0.02
        self.ppo_value_coef = 0.5
        self.ppo_lr = 3e-4
        self.ppo_hidden = 64
        self.ppo_rollout_len = 16
        self.ppo_epochs = 2
        self.ppo_minibatch_size = 64
        self.ppo_max_grad_norm = 0.5
        self.mappo_eval = False
        self.mappo_dpp_reward = True
        self.mappo_gamma = 0.99
        self.mappo_gae_lambda = 0.95
        self.mappo_clip_eps = 0.2
        self.mappo_entropy_coef = 0.02
        self.mappo_value_coef = 0.5
        self.mappo_lr = 3e-4
        self.mappo_hidden = 64
        self.mappo_rollout_len = 16
        self.mappo_epochs = 2
        self.mappo_minibatch_size = 128
        self.mappo_max_grad_norm = 0.5

        self.action_freq = np.zeros(
            [self.user_num, len(self.models), self.top_l_cells])

    def _init_action_and_gp(self):
        """Joint (model, cell_rank) action space and per-user GP optimizers."""
        self.action = {
            "model": np.arange(len(self.models)),
            "cell_rank": np.arange(self.top_l_cells),
        }
        self.contexts = {
            'delay_constraint': '',
            'energy_constraint': '',
            'transmission_rate': '',
            "energy_weight": '',
            "delay_weight": ''
        }
        self.action_dim = len(self.action)
        self.context_dim = len(self.contexts)

        self.length_scale = np.ones(self.context_dim + self.action_dim)
        self.kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=self.length_scale)

        self.noise = 1e-6
        self.beta_function = 'const'
        self.beta_const_val = 2.5
        # Reward-GP random burn-in so UCB/DTS start exploratory (learn reward, not Acc table).
        self.gp_init_random = int(getattr(self, 'gp_init_random', 15))

        self.optimizers = {
            user: cbo.ContextualBayesianOptimization(
                all_actions_dict=self.action,
                contexts=self.contexts,
                kernel=self.kernel,
                init_random=self.gp_init_random,
            ) for user in self.users
        }

    def update_users(self, new_user_num):
        """Update user number and reinitialize dependent variables."""
        np.random.seed(self.seed)  # Set random seed for reproducibility
        self.user_num = new_user_num
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]  # Generate user names

        # Mobile device (MD) parameters for each user (take the first N users' parameters)

        self.md_params = {
            user: {
                'freq': self.md_params_origin[user]['freq'],  # Take the first N GPU frequencies
                'cores': self.md_params_origin[user]['cores'],  # Take the first N GPU core counts
                'flops_per_cycle': self.md_params_origin[user]['flops_per_cycle'],  # Take the first N FLOPs per GPU cycle
                'power_coeff': self.md_params_origin[user]['power_coeff'],
                # Take the first N power consumption coefficients
                'trans_power': self.md_params_origin[user]['trans_power']  # Transmission power remains constant
            } for user in self.users  # Only process the first N users
        }

        # Edge server (ES) parameters (take the first N values)
        self.es_params = {
            'freq': self.es_params_origin['freq'],  # Take the first N GPU frequencies
            'cores': self.es_params_origin['cores'],  # Take the first N GPU core counts
            'flops_per_cycle': self.es_params_origin['flops_per_cycle'],  # Take the first N FLOPs per GPU cycle
            'power_coeff': self.es_params_origin['power_coeff']  # Take the first N power consumption coefficients
        }

        # Fixed parameters for delay, energy, and energy weight (take the first N users)
        self.fixed_delay = {user: self.fixed_delay_origin[user] for user in self.users}  # Only for the first N users
        self.fixed_energy = {user: self.fixed_energy_origin[user] for user in self.users}  # Only for the first N users
        self.fixed_energy_weight = {user: self.fixed_energy_weight_origin[user] for user in
                                    self.users}  # Only for the first N users
        self.arrival_rate = {user: self.arrival_rate_origin[user] for user in self.users}
        self.energy_budget = {user: self.energy_budget_origin[user] for user in self.users}
        # Instantaneous performance metrics tracking
        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [], "vio_degree": [], "reward": [],
                   "mcs": [], "bler": [], "backlog": [], "energy_queue": [], "arrivals": [], "served": [],
                   "cell": []}
            for user in self.users
        }
        self.rewards_history = {user: [] for user in self.users}  # History of rewards for users

        from omnis.sinr_trace import SinrTrace
        # Nested fairness: always a prefix of the fixed max UE pool.
        if not hasattr(self, "sinr_ue_pool") or len(self.sinr_ue_pool) < self.user_num:
            from omnis.sinr_trace import select_spread_ue_ids
            _probe = SinrTrace.from_config_dir(
                self.sinr_trace_dir, tag=self.sinr_trace_tag)
            pool_n = max(self.user_num, int(getattr(self, "ue_pool_size", self.user_num)))
            self.sinr_ue_pool = select_spread_ue_ids(
                pool_n, _probe.num_ues, _probe.num_cells)
            self.ue_pool_size = pool_n
        self.sinr_trace = SinrTrace.from_config_dir(
            self.sinr_trace_dir, tag=self.sinr_trace_tag,
            ue_ids=self.sinr_ue_pool[:self.user_num])
        self._init_action_and_gp()

        # Track action selection frequencies [user, model, cell_rank]
        self.action_freq = np.zeros([self.user_num, len(self.models), self.top_l_cells])