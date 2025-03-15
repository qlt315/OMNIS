import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
import omnis.cbo as cbo  # Assuming ContextualBayesianOptimization is part of cbo module
import omnis.util as util  # Assuming UtilityFunction is part of util module


class Config:
    def __init__(self,seed):
        # Basic configuration
        self.seed = seed
        np.random.seed(self.seed)  # Set random seed for reproducibility

        # Model configuration: Different quantization methods and channels
        self.models = [
            {'name': f'{q}{c}', 'quant_method': q, 'quant_channel': c}
            for q, c in [('Box', 3), ('Box', 6), ('Box', 12), ('Standard', 3), ('Standard', 6), ('Standard', 12)]
        ]

        # Fixed data sizes for each model (in bytes)
        self.data_size = {
            'Box3': 6e3, 'Box6': 13.62e3, 'Box12': 33.58e3,
            'Standard3': 11.23e3, 'Standard6': 22.46e3, 'Standard12': 44.93e3
        }

        # Floating-point operations (FLOPs) for model processing
        self.head_flops = {
            'Box3': 5e10, 'Box6': 1e10, 'Box12': 1.5e10,
            'Standard3': 6e10, 'Standard6': 1.1e10, 'Standard12': 1.4e9
        }
        self.tail_flops = {
            'Box3': 1.2e10, 'Box6': 1.3e10, 'Box12': 1.8e10,
            'Standard3': 1.5e10, 'Standard6': 1.8e10, 'Standard12': 2.0e10
        }

        # User-specific configurations
        self.time_slot_num = 150 # Number of time slots
        self.user_num = 5  # Number of users
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]  # Generate user names

        # Mobile device (MD) parameters for each user
        self.md_params = {user: {
            'freq': np.random.uniform(0.8, 1),  # Randomized CPU frequency
            'cores': np.random.randint(100, 200),  # Number of CPU cores
            'flops_per_cycle': np.random.randint(1, 5),  # FLOPs per CPU cycle
            'power_coeff': np.random.uniform(0.1, 0.3),  # Power consumption coefficient
            'trans_power': 0.1  # Transmission power (W)
        } for user in self.users}

        # Edge server (ES) parameters
        self.es_params = {'freq': 1.5, 'cores': 2048, 'flops_per_cycle': 2, 'power_coeff': 0.7}

        # Channel coding rates and constraints
        self.available_coding_rate = [1, 1 / 2, 1 / 5]
        self.fixed_delay = {user: np.random.uniform(0.5, 1) for user in self.users}  # Delay constraints
        self.fixed_energy = {user: np.random.uniform(0.1, 1.0) for user in self.users}  # Energy constraints
        self.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in self.users}  # Energy weight factors

        # Instantaneous performance metrics tracking
        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [], "vio_degree": [], "reward": []} for user in
            self.users
        }

        # Communication parameters
        self.total_bandwidth = 1e6  # Total available bandwidth (Hz)
        self.est_err = 0.5  # Estimation error for transmission
        self.noise_power_dBm = -174 + 10 * np.log10(self.total_bandwidth)  # Convert dBm to linear scale
        self.noise_power = 10 ** (self.noise_power_dBm / 10 - 3)  # Compute noise power

        # Load precomputed accuracy data
        self.acc_data = pd.read_csv("sys_data/acc_data/fitted_acc_data.csv")

        # Load MIMO channel data
        self.channel_data = np.load("sys_data/trans_sys_sim/mimo_channel_gen/mimo_channel_data.npy")

        # Optimization parameters
        self.bcd_flag = 10e-5  # Convergence threshold for BCD
        self.bcd_max_iter = 30  # Maximum iterations for BCD
        self.average_metrics = {}  # Placeholder for aggregated performance metrics
        self.std_metrics = {}
        self.rewards_history = {user: [] for user in self.users}  # History of rewards for users

        # Action and context definitions for Gaussian Process (GP) models
        self.action = {"model": np.arange(len(self.models))}  # Available actions (model selection)
        self.contexts = {
            'delay_constraint': '',
            'energy_constraint': '',
            'transmission_rate': '',
            "energy_weight": '',
            "delay_weight": ''
        }
        self.action_dim = len(self.action)  # Dimension of action space
        self.context_dim = len(self.contexts)  # Dimension of context space

        # Gaussian Process (GP) kernel settings
        self.length_scale = np.ones(self.context_dim + self.action_dim)  # Length scale for kernel function
        self.kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=self.length_scale)  # GP kernel

        # GP noise and acquisition function settings
        self.noise = 1e-6  # Small noise for GP stability
        self.beta_function = 'const'  # Type of exploration-exploitation tradeoff function
        self.beta_const_val = 2.5  # Constant beta value for exploration

        # Bayesian optimization setup for each user
        self.optimizers = {
            user: cbo.ContextualBayesianOptimization(
                all_actions_dict=self.action,
                contexts=self.contexts,
                kernel=self.kernel
            ) for user in self.users
        }

        # Utility function for decision-making (Upper Confidence Bound - UCB)

        self.utility = util.UtilityFunction(kind="ucb", beta_kind=self.beta_function, beta_const=self.beta_const_val)

        # Track action selection frequencies
        self.action_freq = np.zeros([self.user_num, len(self.models)])

    def reset_seed(self):
        """Reset the random seed for reproducibility."""
        np.random.seed(self.seed)

    def update_users(self, new_user_num):
        """Update user number and reinitialize dependent variables."""
        self.reset_seed()  # Ensure consistency when updating users
        self.user_num = new_user_num
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]  # Generate user names

        # Mobile device (MD) parameters for each user
        self.md_params = {user: {
            'freq': np.random.uniform(0.8, 1),  # Randomized CPU frequency
            'cores': np.random.randint(100, 200),  # Number of CPU cores
            'flops_per_cycle': np.random.randint(1, 5),  # FLOPs per CPU cycle
            'power_coeff': np.random.uniform(0.1, 0.3),  # Power consumption coefficient
            'trans_power': 0.1  # Transmission power (W)
        } for user in self.users}
        self.fixed_delay = {user: np.random.uniform(0.5, 1) for user in self.users}  # Delay constraints
        self.fixed_energy = {user: np.random.uniform(0.1, 1.0) for user in self.users}  # Energy constraints
        self.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in
                                    self.users}  # Energy weight factors

        # Instantaneous performance metrics tracking
        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [], "vio_degree": [], "reward": []} for user
            in
            self.users
        }
        self.rewards_history = {user: [] for user in self.users}  # History of rewards for users
        # Bayesian optimization setup for each user
        self.optimizers = {
            user: cbo.ContextualBayesianOptimization(
                all_actions_dict=self.action,
                contexts=self.contexts,
                kernel=self.kernel
            ) for user in self.users
        }
        # Track action selection frequencies
        self.action_freq = np.zeros([self.user_num, len(self.models)])