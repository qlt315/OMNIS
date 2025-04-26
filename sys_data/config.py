import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
import omnis.cbo as cbo
import omnis.util as util

class Config:
    def __init__(self, seed):
        # Basic configuration
        self.seed = seed
        np.random.seed(self.seed)
        self.time_slot_num = 200

        # Model configuration
        self.models = [
            {'name': f'{q}{c}', 'quant_method': q, 'quant_channel': c}
            for q, c in [('Box', 3), ('Box', 6), ('Box', 12), 
                        ('Standard', 3), ('Standard', 6), ('Standard', 12)]
        ]

        # Data sizes and computational requirements
        self.data_size = {
            'Box3': 6e3, 'Box6': 13.26e3, 'Box12': 33.58e3,
            'Standard3': 11.23e3, 'Standard6': 22.46e3, 'Standard12': 44.93e3
        }
        
        # Updated FLOPs for modern hardware
        self.head_flops = {
            'Box3': 2.585e12, 'Box6': 2.585e12, 'Box12': 2.585e12,
            'Standard3': 2.585e12, 'Standard6': 2.585e12, 'Standard12': 2.585e12
        }
        self.tail_flops = {
            'Box3': 34672e9, 'Box6': 34672e9, 'Box12': 34672e9,
            'Standard3': 34672e9, 'Standard6': 34672e9, 'Standard12': 34672e9
        }

        # User configuration
        self.user_num = 8
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]

        # Device parameters - Updated for modern mobile GPUs
        self.md_params = {user: {
            'freq': np.random.uniform(1.2, 2.0),  # GPU freq in GHz
            'cores': np.random.randint(512, 1024),  # GPU cores
            'flops_per_cycle': np.random.randint(8, 16),
            'power_coeff': np.random.uniform(0.2, 0.5),
            'trans_power': 0.1  # Transmission power in W
        } for user in self.users}

        # Edge server parameters - Updated for modern server GPUs
        self.es_params = {
            'freq': np.random.uniform(3, 5),  # Server GPU freq in GHz
            'cores': np.random.randint(8192, 16384),
            'flops_per_cycle': np.random.randint(16, 32),
            'power_coeff': np.random.uniform(0.5, 1.0)
        }

        # Store original parameters for scaling
        self.md_params_origin = self.md_params.copy()
        self.es_params_origin = self.es_params.copy()

        # Communication parameters
        self.available_coding_rate = [1, 1/2, 1/5]
        self.total_bandwidth = 1e5  # Hz
        self.est_err = 0.5
        self.noise_power_dBm = -174 + 10 * np.log10(self.total_bandwidth)
        self.noise_power = 10 ** (self.noise_power_dBm / 10 - 3)

        # Constraints
        self._initialize_constraints()

        # Performance tracking
        self._initialize_metrics()

        # Load data
        self._load_precomputed_data()

        # Optimization parameters
        self._initialize_optimization()

    def _initialize_constraints(self):
        """Initialize system constraints"""
        self.fixed_delay = {user: np.random.uniform(0.8, 1.5) for user in self.users}
        self.fixed_energy = {user: np.random.uniform(0.8, 1.5) for user in self.users}
        self.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in self.users}
        
        # Store originals for scaling
        self.fixed_delay_origin = self.fixed_delay.copy()
        self.fixed_energy_origin = self.fixed_energy.copy()
        self.fixed_energy_weight_origin = self.fixed_energy_weight.copy()

    def _initialize_metrics(self):
        """Initialize performance metrics tracking"""
        self.instant_metrics = {
            user: {
                "delay": [], "energy": [], "accuracy": [], 
                "is_vio": [], "vio_degree": [], "reward": []
            } for user in self.users
        }
        self.rewards_history = {user: [] for user in self.users}
        self.average_metrics = {}
        self.std_metrics = {}
        self.action_freq = np.zeros([self.user_num, len(self.models)])

    def _load_precomputed_data(self):
        """Load precomputed data files"""
        try:
            self.acc_data = pd.read_csv("sys_data/acc_data/fitted_acc_data.csv")
            self.channel_data = np.load("sys_data/mimo_channel_gen/mimo_channel_data.npy")
        except FileNotFoundError as e:
            print(f"Error loading precomputed data: {e}")
            raise

    def _initialize_optimization(self):
        """Initialize optimization parameters"""
        # BCD parameters
        self.bcd_flag = 10e-5
        self.bcd_max_iter = 30

        # Action and context spaces
        self.action = {"model": np.arange(len(self.models))}
        self.contexts = {
            'delay_constraint': '',
            'energy_constraint': '',
            'transmission_rate': '',
            "energy_weight": '',
            "delay_weight": ''
        }
        self.action_dim = len(self.action)
        self.context_dim = len(self.contexts)

        # GP parameters
        self.length_scale = np.ones(self.context_dim + self.action_dim)
        self.kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=self.length_scale)
        self.noise = 1e-6
        self.beta_function = 'const'
        self.beta_const_val = 2.5

        # Initialize optimizers and utility function
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Setup optimization components"""
        self.optimizers = {
            user: cbo.ContextualBayesianOptimization(
                all_actions_dict=self.action,
                contexts=self.contexts,
                kernel=self.kernel
            ) for user in self.users
        }
        self.utility = util.UtilityFunction(
            kind="ucb",
            beta_kind=self.beta_function,
            beta_const=self.beta_const_val
        )

    def update_users(self, new_user_num):
        """Update system for new number of users"""
        self.user_num = new_user_num
        self.users = [f'user_{i + 1}' for i in range(self.user_num)]
        
        # Update all user-dependent parameters
        self._update_device_params()
        self._update_constraints()
        self._initialize_metrics()
        self._setup_optimizers()

    def _update_device_params(self):
        """Update device parameters for new user count"""
        self.md_params = {
            user: {k: self.md_params_origin[user][k] 
                  for k in self.md_params_origin[user]}
            for user in self.users
        }
        self.es_params = self.es_params_origin.copy()

    def _update_constraints(self):
        """Update constraints for new user count"""
        self.fixed_delay = {
            user: self.fixed_delay_origin[user] 
            for user in self.users
        }
        self.fixed_energy = {
            user: self.fixed_energy_origin[user] 
            for user in self.users
        }
        self.fixed_energy_weight = {
            user: self.fixed_energy_weight_origin[user] 
            for user in self.users
        }