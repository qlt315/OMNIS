import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
import omnis.cbo as cbo  # Assuming ContextualBayesianOptimization is part of cbo module
import omnis.util as util  # Assuming UtilityFunction is part of util module
seed = 42
np.random.seed(seed)

class Config:
    def __init__(self):
        # Models configuration
        self.models = [
            {'name': f'{q}{c}', 'quant_method': q, 'quant_channel': c}
            for q, c in [('Box', 3), ('Box', 6), ('Box', 12), ('Standard', 3), ('Standard', 6), ('Standard', 12)]
        ]

        # Fixed data sizes for each model
        self.data_size = {
            'Box3': 6e3, 'Box6': 13.62e3, 'Box12': 33.58e3,
            'Standard3': 11.23e3, 'Standard6': 22.46e3, 'Standard12': 44.93e3
        }

        # Fixed FLOPs for head and tail
        self.head_flops = {'Box3': 5e10, 'Box6': 1e10, 'Box12': 1.5e10, 'Standard3': 6e10, 'Standard6': 1.1e10,
                           'Standard12': 1.4e9}
        self.tail_flops = {'Box3': 1.2e10, 'Box6': 1.3e10, 'Box12': 1.8e10, 'Standard3': 1.5e10, 'Standard6': 1.8e10,
                           'Standard12': 2.0e10}

        # User-specific initialization
        self.time_slot_num = 150
        self.user_num = 5
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]
        self.md_params = {user: {
            'freq': np.random.uniform(0.8, 1),
            'cores': np.random.randint(100, 200),
            'flops_per_cycle': np.random.randint(1, 5),
            'power_coeff': np.random.uniform(0.1, 0.3),
            'trans_power': 0.1
        } for user in self.users}

        self.es_params = {'freq': 1.5, 'cores': 2048, 'flops_per_cycle': 2, 'power_coeff': 0.7}

        # Channel coding rates and other metrics
        self.available_coding_rate = [1, 1 / 2, 1 / 5]
        self.fixed_delay = {user: np.random.uniform(0.5, 1) for user in self.users}
        self.fixed_energy = {user: np.random.uniform(0.1, 1.0) for user in self.users}
        self.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in self.users}
        self.instant_metrics = {
            user: {"delay": [], "energy": [], "accuracy": [], "is_vio": [], "vio_degree": [], "reward": []} for user in self.users
        }

        # Total bandwidth and noise power calculations
        self.total_bandwidth = 1e6
        self.est_err = 0.5
        self.noise_power_dBm = -174 + 10 * np.log10(self.total_bandwidth)
        self.noise_power = 10 ** (self.noise_power_dBm / 10 - 3)

        # Load accuracy data
        self.acc_data = pd.read_csv("sys_data/acc_data/fitted_acc_data.csv")

        # Optimization and rewards
        self.bcd_flag = 10e-5
        self.bcd_max_iter = 30
        self.average_metrics = {}
        self.rewards_history = {user: [] for user in self.users}

        # Action and context for GP models
        self.action = {"model": np.array([0, 1, 2, 3, 4, 5])}
        self.contexts = {
            'delay_constraint': '',
            'energy_constraint': '',
            'transmission_rate': '',
            "energy_weight": '',
            "delay_weight": ''
        }
        self.action_dim = len(self.action)
        self.context_dim = len(self.contexts)

        # GP kernel settings
        self.length_scale = np.ones(self.context_dim + self.action_dim)
        self.kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=self.length_scale)

        # Noise and beta function
        self.noise = 1e-6
        self.beta_function = 'const'
        self.beta_const_val = 2.5

        # Create optimizer for each user
        self.optimizers = {
            user: cbo.ContextualBayesianOptimization(
                all_actions_dict=self.action,
                contexts=self.contexts,
                kernel=self.kernel
            ) for user in self.users
        }

        # Utility function (shared among users)
        self.utility = util.UtilityFunction(kind="ucb", beta_kind=self.beta_function, beta_const=self.beta_const_val)
