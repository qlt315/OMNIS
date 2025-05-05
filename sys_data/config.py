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


        # Channel coding rates and constraints
        self.available_coding_rate = [1, 1 / 2, 1 / 5]
        self.fixed_delay = {user: np.random.uniform(0.8, 1.5) for user in self.users}  # Delay constraints
        self.fixed_energy = {user: np.random.uniform(0.8, 1.5) for user in self.users}  # Energy constraints
        self.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in self.users}  # Energy weight factors

        self.fixed_delay_origin = self.fixed_delay
        self.fixed_energy_origin = self.fixed_energy
        self.fixed_energy_weight_origin = self.fixed_energy_weight

        # Instantaneous performance metrics tracking
        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [], "vio_degree": [], "reward": []} for user in
            self.users
        }

        # Communication parameters
        self.total_bandwidth = 1e5  # Total available bandwidth (Hz)
        self.est_err = 0.5  # Estimation error for transmission
        self.noise_power_dBm = -174 + 10 * np.log10(self.total_bandwidth)  # Convert dBm to linear scale
        self.noise_power = 10 ** (self.noise_power_dBm / 10 - 3)  # Compute noise power

        # Load precomputed accuracy data
        self.acc_data = pd.read_csv("sys_data/acc_data/fitted_acc_data.csv")

        # Load MIMO channel data
        self.channel_data = np.load("sys_data/mimo_channel_gen/mimo_channel_data.npy")

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

        # Add distribution configuration
        self.distribution = 'laplace'  # or 'gaussian'
        self.laplace_scale = 2.0

        # Update optimizer configuration
        self.optimizer_params = {
            'distribution': self.distribution,
            'laplace_scale': self.laplace_scale
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