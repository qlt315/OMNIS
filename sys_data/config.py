import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
import omnis.cbo as cbo  # Assuming ContextualBayesianOptimization is part of cbo module
import omnis.util as util  # Assuming UtilityFunction is part of util module
from omnis.mcs_table import McsTable


class Config:
    def __init__(self,seed):
        # Basic configuration
        self.seed = seed
        np.random.seed(self.seed)  # Set random seed for reproducibility
        self.time_slot_num = 200 # Number of time slots

        # Model configuration: Different quantization methods and channels
        self.models = [
            {'name': f'{q}{c}', 'quant_method': q, 'quant_channel': c}
            for q, c in [('Box', 3), ('Box', 6), ('Box', 12), ('Standard', 3), ('Standard', 6), ('Standard', 12)]
        ]

        # Fixed data sizes for each model (in bytes)
        self.data_size = {
            'Box3': 6e3, 'Box6': 13.26e3, 'Box12': 33.58e3,
            'Standard3': 11.23e3, 'Standard6': 22.46e3, 'Standard12': 44.93e3
        }

        # Floating-point operations (FLOPs) for model processing
        self.head_flops = {
            'Box3': 2.585e12, 'Box6': 2.585e12, 'Box12': 2.585e12,
            'Standard3': 2.585e12, 'Standard6': 2.585e12, 'Standard12': 2.585e12
        }
        self.tail_flops = {
            'Box3': 34672e9, 'Box6': 34672e9, 'Box12': 34672e9,
            'Standard3': 34672e9, 'Standard6': 34672e9, 'Standard12': 34672e9
        }


        # # Floating-point operations (FLOPs) for model processing
        # self.head_flops = {
        #     'Box3': 5e10, 'Box6': 1e10, 'Box12': 1.5e10,
        #     'Standard3': 6e10, 'Standard6': 1.1e10, 'Standard12': 1.4e9
        # }
        # self.tail_flops = {
        #     'Box3': 1.2e10, 'Box6': 1.3e10, 'Box12': 1.8e10,
        #     'Standard3': 1.5e10, 'Standard6': 1.8e10, 'Standard12': 2.0e10
        # }


        # User-specific configurations
        self.user_num = 8  # Number of users
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]  # Generate user names

        # Mobile device (MD) parameters for each user
        self.md_params = {user: {
            'freq': np.random.uniform(1.2, 2.0),  # GPU frequency (closer to real mobile GPUs)
            'cores': np.random.randint(512, 1024),  # Number of GPU cores
            'flops_per_cycle': np.random.randint(8, 16),  # FLOPs per GPU cycle
            'power_coeff': np.random.uniform(0.2, 0.5),  # Power consumption coefficient
            'trans_power': 0.1  # Transmission power (W)
        } for user in self.users}


        # Edge server (ES) parameters
        self.es_params = {
            'freq': np.random.uniform(3, 5),  # GPU frequency
            'cores': np.random.randint(8192, 16384),  # number of GPU cores
            'flops_per_cycle': np.random.randint(16, 32),  # FLOPs per GPU cycle
            'power_coeff': np.random.uniform(0.5, 1.0)  # Adjusted power consumption coefficient
        }

        self.md_params_origin = self.md_params
        self.es_params_origin = self.es_params

        # # Mobile device (MD) parameters for each user
        # self.md_params = {user: {
        #     'freq': np.random.uniform(0.8, 1),  # Randomized CPU frequency
        #     'cores': np.random.randint(100, 200),  # Number of CPU cores
        #     'flops_per_cycle': np.random.randint(1, 5),  # FLOPs per CPU cycle
        #     'power_coeff': np.random.uniform(0.1, 0.3),  # Power consumption coefficient
        #     'trans_power': 0.1  # Transmission power (W)
        # } for user in self.users}
        #
        # # Edge server (ES) parameters
        # self.es_params = {'freq': 1.5, 'cores': 2048, 'flops_per_cycle': 2, 'power_coeff': 0.7}


        # Task constraints (loosened vs conference: sojourn delay = queue wait +
        # service, so the deadline must leave headroom for backlog drain)
        self.fixed_delay = {user: np.random.uniform(1.5, 2.5) for user in self.users}  # Delay constraints
        self.fixed_energy = {user: np.random.uniform(0.8, 1.5) for user in self.users}  # Energy constraints
        self.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in self.users}  # Energy weight factors

        # ---- Queueing model + Lyapunov framework (journal extension) ----
        self.slot_duration = 1.0  # Slot length [s]; service = bandwidth * SE * slot_duration
        # Per-user Poisson task arrival rate [tasks/slot]
        # Arrival rate tuned to the "tight but feasible" regime: queues stabilize
        # at a small POSITIVE backlog (not 0, not diverging). backlog=0 means the
        # link is over-provisioned and erases inter-algorithm differences; a
        # small positive backlog is the DPP-optimal operating point (Little's law:
        # just-enough delay, max reward). [0.4, 0.65] empirically settles to a
        # small positive steady-state backlog under the feasible energy budget.
        self.arrival_rate = {user: np.random.uniform(0.4, 0.65) for user in self.users}
        # Per-user average energy budget [J/slot] for the virtual energy queue.
        # Set to 1.10 (above the cheapest feasible service ~0.9-1.0 J) so the
        # energy constraint is FEASIBLE: an infeasible budget makes the virtual
        # queue Z diverge, its drift term explodes and drowns the reward signal.
        self.energy_budget = {user: 1.10 for user in self.users}
        self.arrival_rate_origin = self.arrival_rate
        self.energy_budget_origin = self.energy_budget
        # Lyapunov penalty weight V: trades time-average reward against queue drift
        self.lyapunov_v = 1.0
        # Normalization scales so the drift terms are O(1) against the reward
        self.dpp_bit_scale = float(np.mean(list(self.data_size.values())))  # ~2.2e4 bits
        self.dpp_energy_scale = 0.80  # J

        self.fixed_delay_origin = self.fixed_delay
        self.fixed_energy_origin = self.fixed_energy
        self.fixed_energy_weight_origin = self.fixed_energy_weight

        # Instantaneous performance metrics tracking
        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [], "vio_degree": [], "reward": [],
                   "mcs": [], "bler": [], "backlog": [], "energy_queue": [], "arrivals": [], "served": [],
                   "cell": []}
            for user in self.users
        }

        # Communication parameters
        self.total_bandwidth = 1e5  # Total available bandwidth (Hz) per cell
        self.est_err = 0.5  # Legacy (unused with SINR traces)
        self.est_err_db = 1.0  # SINR estimation noise std [dB] on trace values
        self.noise_power_dBm = -174 + 10 * np.log10(self.total_bandwidth)  # Convert dBm to linear scale
        self.noise_power = 10 ** (self.noise_power_dBm / 10 - 3)  # Compute noise power

        # Multi-cell PHY: array-MIMO SINR traces (joint model, cell_rank arms)
        self.num_cells = 7
        self.top_l_cells = 3
        self.sinr_trace_tag = "smoke7"
        self.sinr_trace_dir = "phy_sim/output"
        # Spread system users across distinct hex sites (site = 3 co-sited
        # sectors; each site drops 3*ues_per_cell UEs). Picking one UE per site
        # (stride = UEs-per-site) gives a realistic multi-cell user spread rather
        # than clustering all users into site 0.
        from omnis.sinr_trace import SinrTrace
        _probe = SinrTrace.from_config_dir(self.sinr_trace_dir, tag=self.sinr_trace_tag)
        _ues_per_site = max(1, _probe.num_ues // _probe.num_cells)
        _spread = [min(i * _ues_per_site, _probe.num_ues - 1)
                   for i in range(self.user_num)]
        self.sinr_trace = SinrTrace.from_config_dir(
            self.sinr_trace_dir, tag=self.sinr_trace_tag, ue_ids=_spread)

        # PHY layer: Sionna-generated MCS tables (mcs_def / acc / bler / acc_clean)
        self.mcs_table = McsTable("phy_sim/output", acc_floor=0.0, bler_target=0.1)
        self.available_mcs = list(range(self.mcs_table.num_mcs))
        self.bler_target = self.mcs_table.bler_target  # ILLA operating point

        # Legacy MIMO channel cube (kept for backwards compat; unused with traces)
        self.channel_data = np.load("sys_data/mimo_channel_gen/mimo_channel_data.npy")

        # Optimization parameters
        self.bcd_flag = 10e-5  # Convergence threshold for BCD
        self.bcd_max_iter = 30  # Maximum iterations for BCD
        self.average_metrics = {}  # Placeholder for aggregated performance metrics
        self.std_metrics = {}
        self.rewards_history = {user: [] for user in self.users}  # History of rewards for users

        # Action and context definitions for Gaussian Process (GP) models
        self._init_action_and_gp()

        # Utility function for decision-making (Upper Confidence Bound - UCB)

        self.utility = util.UtilityFunction(kind="ucb", beta_kind=self.beta_function, beta_const=self.beta_const_val)

        # Algorithm selection: 'ucb' (conference version) or 'causal' (journal extension)
        self.algo = 'causal'
        # Std of the per-task accuracy observation noise, simulating per-image
        # accuracy fluctuations around the pre-computed accuracy curves (0 disables)
        self.acc_noise_std = 0.02
        # Causal MAB settings
        self.causal_acq = 'ucb'            # Acquisition function: 'ucb' or 'ts'
        self.causal_use_prior = True       # Use the offline causal prior mean
        self.causal_shared = True          # Pool the accuracy mechanism GP across MDs
        self.causal_prior_snr_step = 10    # Coarseness of the offline prior SNR grid
        # Residual GP hyperparameters over (snr_db, quant_flag, channels, mcs_index)
        self.causal_gp_length_scales = [2.0, 0.75, 2.0, 2.0]
        self.causal_gp_signal_var = 2.5e-3
        # CTO joint space is (6*L)^U; cap candidates per suggest() call
        self.cto_max_candidates = 2000
        # GDO = SEM-O-RAN SF-ESP greedy (Puligheddu et al., TMC 2024)
        # Stricter Ac → heavier z*; myopic GDO underperforms Causal on DPP.
        self.gdo_acc_floor = 0.25
        self.gdo_ref_snr_db = 5.0
        self.gdo_ref_mcs = None
        self.gdo_price_radio = 1.0
        self.gdo_price_compute = 1.0
        self.gdo_offer_scale = 1.0
        # Shared RL flags
        self.rl_dpp_reward = True
        self.rl_eval = False
        # Centralized branching DQN
        self.dqn_pretrain_slots = 400
        self.dqn_model_path = "figures/dqn_pretrained.pt"
        self.dqn_eval = False
        self.dqn_dpp_reward = True
        self.dqn_gamma = 0.95
        self.dqn_batch_size = 32
        self.dqn_buffer_size = 8000
        self.dqn_lr = 1e-3
        self.dqn_target_sync = 20
        self.dqn_eps_start = 1.0
        self.dqn_eps_end = 0.05
        self.dqn_eps_decay_slots = 300
        self.dqn_train_every = 1
        self.dqn_grad_steps = 4
        self.dqn_hidden = 128
        # MAPPO (CTDE)
        self.mappo_eval = False
        self.mappo_dpp_reward = True
        self.mappo_gamma = 0.95
        self.mappo_gae_lambda = 0.95
        self.mappo_clip_eps = 0.2
        self.mappo_entropy_coef = 0.01
        self.mappo_value_coef = 0.5
        self.mappo_lr = 3e-4
        self.mappo_hidden = 128
        self.mappo_rollout_len = 32
        self.mappo_epochs = 4
        self.mappo_minibatch_size = 64
        self.mappo_max_grad_norm = 0.5

        # Track action selection frequencies [user, model, cell_rank]
        self.action_freq = np.zeros([self.user_num, len(self.models), self.top_l_cells])

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

        self.optimizers = {
            user: cbo.ContextualBayesianOptimization(
                all_actions_dict=self.action,
                contexts=self.contexts,
                kernel=self.kernel
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
        _probe = SinrTrace.from_config_dir(self.sinr_trace_dir, tag=self.sinr_trace_tag)
        _ues_per_site = max(1, _probe.num_ues // _probe.num_cells)
        _spread = [min(i * _ues_per_site, _probe.num_ues - 1)
                   for i in range(self.user_num)]
        self.sinr_trace = SinrTrace.from_config_dir(
            self.sinr_trace_dir, tag=self.sinr_trace_tag, ue_ids=_spread)
        self._init_action_and_gp()

        # Track action selection frequencies [user, model, cell_rank]
        self.action_freq = np.zeros([self.user_num, len(self.models), self.top_l_cells])