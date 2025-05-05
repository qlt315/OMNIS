import numpy as np
import time
from scipy.special import erf
from scipy.stats import laplace
import random
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sys_data.config import Config
import omnis.util as util  # Assuming UtilityFunction is part of util module
from omnis.causal_model import CausalModel
from omnis.causal_inference import CausalInference
import networkx as nx


class DTS:
    def __init__(self, config):
        """Initialize system parameters including models, devices, and users."""
        # Basic information
        self.name = "dts"
        self.seed = config.seed
        np.random.seed(self.seed)  # Ensure reproducibility

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
        self.available_coding_rate = config.available_coding_rate
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
        self.acc_data = config.acc_data
        self.channel_data = config.channel_data
        self.est_err = config.est_err
        self.action_freq = config.action_freq

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
        self.setup_causal_model()

        # Laplacian noise parameter
        self.laplace_scale = 2.0  # Controls noise magnitude

    def setup_causal_model(self):
        """Initialize causal model components"""
        # Create causal graph
        G = nx.DiGraph()

        # Add nodes for key variables
        nodes = ['SNR', 'Model', 'CodingRate', 'Accuracy', 'Delay', 'Energy']
        G.add_nodes_from(nodes)

        # Add edges representing causal relationships
        edges = [
            ('SNR', 'Accuracy'),
            ('SNR', 'Delay'),
            ('Model', 'Accuracy'),
            ('Model', 'Delay'),
            ('Model', 'Energy'),
            ('CodingRate', 'Accuracy'),
            ('CodingRate', 'Delay')
        ]
        G.add_edges_from(edges)

        # Initialize causal inference and model
        self.causal_inference = CausalInference(G)
        self.causal_model = CausalModel()

    def generate_tasks(self, time_slot):
            """Dynamically adjust delay and energy constraints while keeping the base values fixed."""

            task_dic = {
                user: {
                    "delay_constraint": self.fixed_delay[user]+ np.random.uniform(-0.001, 0.001),
                    "energy_constraint": self.fixed_energy[user]+ np.random.uniform(-0.001, 0.001),
                    "energy_weight": self.fixed_energy_weight[user]+ np.random.uniform(-0.001, 0.001),
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

    def model_selection(self, context_dic):
        """Enhanced model selection with Laplacian noise"""
        model_selection_dic = {}

        for user_idx, user in enumerate(self.users):
            context_m = context_dic[user]
            optimizer_m = self.optimizers[user]
            
            # Get causal effects with Laplacian noise
            snr = context_m.get('transmission_rate', 0)
            causal_effects = self.causal_model.estimate_effect(snr)
            
            if causal_effects is not None:
                # Add Laplacian noise to effects
                noise = laplace.rvs(loc=0, scale=self.laplace_scale, 
                                  size=len(causal_effects))
                causal_effects += noise
            
            action_m = optimizer_m.suggest(context_m, self.utility, 
                                         additional_weights=causal_effects)
            selected_model_m = action_m['model']
            
            self.action_freq[user_idx, selected_model_m] += 1
            model_selection_dic[user] = {
                "model": self.models[selected_model_m]["name"]
            }

        return model_selection_dic

    def update_gp(self, context_dic, model_selection_dic, reward_dic):
        """Update GP and causal model with new observations"""
        for user in self.users:
            optimizer_m = self.optimizers[user]
            context_m = context_dic[user]
            model_m = model_selection_dic[user]['model']
            action_m = next((index for index, model in enumerate(self.models) if model['name'] == model_m), None)
            action_dic_m = {'model': action_m}
            reward_m = reward_dic[user]

            # Update GP
            optimizer_m.register(context_m, action_dic_m, reward_m)

            # Update causal model with actual outcomes
            if hasattr(self, 'causal_model'):
                self.causal_model.observations[-1].update({
                    "Accuracy": self.instant_metrics[user]["accuracy"][-1],
                    "Delay": self.instant_metrics[user]["delay"][-1],
                    "Energy": self.instant_metrics[user]["energy"][-1]
                })

        return

    def get_reward(self, task_dic, acc_dic, total_overhead_dic):
        """Compute the reward of MDs based on accuracy and penalty terms."""
        reward_dic = {}
        for user in self.users:
            reward_dic[user] = (acc_dic[user] + 1.5 * task_dic[user]["delay_weight"] * erf(
                task_dic[user]["delay_constraint"] - total_overhead_dic[user]["delay"])
                                + 1.5 * task_dic[user]["energy_weight"] * erf(
                        task_dic[user]["energy_constraint"] - total_overhead_dic[user]["energy"]))

        return reward_dic

    def get_trans_rate(self, time_slot):
        """Calculate the achievable transmission rate for each user based on channel estimation."""

        # Retrieve the true channel matrix for the current time slot (shape: 10 x 64 x 4)
        true_channel = self.channel_data[time_slot]

        # Channel estimate with error
        random_noise = (np.random.randn(*true_channel.shape) + 1j * np.random.randn(*true_channel.shape)) / np.sqrt(2)
        error_matrix = self.est_err * true_channel * random_noise
        estimated_channel = true_channel + error_matrix

        trans_rate_dic = {user: 0 for user in self.users}  # Initialize transmission rate to 0
        snr_dic = {user: 0 for user in self.users}  # Initialize snr to 0

        for user_idx, user in enumerate(self.users):
            # Compute SNR: SNR_m = ||H_m||^2 / sigma^2
            snr_m = np.linalg.norm(estimated_channel[user_idx, :, :], 'fro') ** 2 / self.noise_power
            snr_dic[user] = snr_m
            # Compute transmission rate using Shannon Capacity Formula
            rate_m = np.log2(1 + snr_m)  # bits per second per Hz
            # Store the transmission rate for the user
            trans_rate_dic[user] = rate_m

        return snr_dic, trans_rate_dic

    def allocate_bandwidth(self, task_dic, model_selection_dic, trans_rate_dic, ldpc_rate_dic):
        """Allocate bandwidth based on the optimization strategy."""
        # Prepare d'_m(t) values for each user
        bandwidth_allocation_dic = {}
        d_prime_dic = {}
        for user in self.users:
            chosen_model_m = model_selection_dic[user]["model"]
            coded_data_size_m = self.data_size[chosen_model_m] * ldpc_rate_dic[user]
            rate_m = trans_rate_dic[user]
            p_m = self.md_params[user]['power_coeff']
            omega_m_t = task_dic[user]['delay_constraint']
            omega_m_e = task_dic[user]['energy_constraint']

            d_prime_dic[user] = coded_data_size_m * (omega_m_t + p_m * omega_m_e) / rate_m

            # Compute the optimal bandwidth allocation b_m*(t) based on the formula
            total_d_prime = sum(np.sqrt(d) for d in d_prime_dic.values())
            bandwidth_allocation_dic[user] = self.total_bandwidth * np.sqrt(d_prime_dic[user]) / total_d_prime

        return bandwidth_allocation_dic

    def gpu_resource_allocation(self, task_dic, model_selection_dic):
        """
        Solve for the optimal GPU frequency allocation `f_m^e(t)`.
        """
        gpu_allocation_dict = {}

        # CVXPY problem definition
        f_m = cp.Variable(self.user_num)  # GPU frequency for M MDs
        # Constraints
        constraints = [0 <= f_m, f_m <= self.es_params['freq'], cp.sum(f_m) <= self.es_params['freq']]

        omega_m_t_vector = np.array([task_dic[user]['delay_constraint'] for user in self.users])
        omega_m_e_vector = np.array([task_dic[user]['energy_constraint'] for user in self.users])
        cores = self.es_params['cores']  # Same for all users
        head_flops_vector = np.array([self.head_flops[model_selection_dic[user]["model"]] for user in self.users])
        flops_per_cycle = self.es_params['flops_per_cycle']  # Same for all users
        freq = self.es_params['freq']
        power_coeff = self.es_params['power_coeff']  # Same for all users

        # Calculate first and second terms for all users in vectorized form
        first_term = omega_m_t_vector * head_flops_vector @ cp.inv_pos(f_m) / (cores * flops_per_cycle)
        second_term = omega_m_e_vector * power_coeff @ (f_m ** 2) / (cores * flops_per_cycle)
        total_sum = cp.sum(first_term * 1e-9 + second_term)

        # Define the objective function to minimize
        objective = cp.Minimize(total_sum)

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)
        # Save the optimal GPU frequencies for each user to the dictionary
        for idx, user in enumerate(self.users):
            gpu_allocation_dict[user] = f_m.value[idx]
        # Return the dictionary containing the optimal GPU frequencies for all users
        return gpu_allocation_dict

    def coding_rate_selection(self, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                              local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic):
        """ Select the optimal channel coding rate for each user. """
        ldpc_rate_dic = {}
        for user in self.users:
            feasible_rates, infeasible_rates = [], []
            feasible_acc, infeasible_penalty = [], []

            for coding_rate in self.available_coding_rate:
                # Temporarily update user's coding rate
                temp_ldpc_rate_dic = {user: random.choice(self.available_coding_rate) for user in self.users}
                temp_ldpc_rate_dic[user] = coding_rate
                acc_dic = self.get_accuracy(snr_dic, temp_ldpc_rate_dic, model_selection_dic)

                # Compute overheads
                edge_overhead_dic = self.get_edge_overhead(model_selection_dic, gpu_allocation_dic)
                trans_overhead_dic = self.get_trans_overhead(trans_rate_dic, model_selection_dic,
                                                             bandwidth_allocation_dic,
                                                             temp_ldpc_rate_dic)

                # Calculate total delay and energy
                total_delay = (local_overhead_dic[user]['delay'] + edge_overhead_dic[user]['delay'] +
                               trans_overhead_dic[user][
                                   'delay'])
                total_energy = (
                        local_overhead_dic[user]['energy'] + edge_overhead_dic[user]['energy'] +
                        trans_overhead_dic[user][
                            'energy'])

                # Check QoS constraints
                if total_delay <= task_dic[user]['delay_constraint'] and total_energy <= task_dic[user][
                    'energy_constraint']:
                    feasible_rates.append(coding_rate)
                    feasible_acc.append(acc_dic[user])
                else:
                    penalty = (acc_dic[user] +
                               task_dic[user]['delay_weight'] * erf(total_delay - task_dic[user]['delay_constraint']) +
                               task_dic[user]['energy_weight'] * erf(
                                total_energy - task_dic[user]['energy_constraint']))
                    infeasible_rates.append(coding_rate)
                    infeasible_penalty.append(penalty)

            # Select best coding rate
            if feasible_rates:
                ldpc_rate_dic[user] = feasible_rates[np.argmax(feasible_acc)]
            else:
                ldpc_rate_dic[user] = infeasible_rates[np.argmin(infeasible_penalty)]

        return ldpc_rate_dic

    def get_accuracy(self, snr_dic, ldpc_rate_dic, model_selection_dic):
        """ Get the interpolated accuracy for a given model, coding rate, and SNR."""

        acc_dic = {}  # Dictionary to store delay and energy consumption for each user
        for user in self.users:
            chosen_model_m = model_selection_dic[user]["model"]  # Get the selected model for this user
            df_filtered = self.acc_data[
                (self.acc_data["Model"] == chosen_model_m) & (self.acc_data["Coding Rate"] == ldpc_rate_dic[user])]

            if df_filtered.empty:
                raise ValueError(f"No data found for Model={chosen_model_m}, CodingRate={ldpc_rate_dic[user]}")

            # Extract SNR and Accuracy values
            snr_values = df_filtered["SNR"].values
            acc_values = df_filtered["Accuracy"].values

            # Ensure SNR is within bounds
            if snr_dic[user] < snr_values.min():
                snr_dic[user] = snr_values.min()  # Set to lower bound if below minimum
            elif snr_dic[user] > snr_values.max():
                snr_dic[user] = snr_values.max()  # Set to upper bound if above maximum

            # Interpolate accuracy for the given SNR
            acc_m = np.interp(snr_dic[user], snr_values, acc_values)
            acc_dic[user] = acc_m

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

    def get_trans_overhead(self, trans_rate_dic, model_selection_dic, bandwidth_allocation_dic, ldpc_rate_dic):
        """Calculate transmission overhead, including delay and energy consumption."""
        trans_overhead_dic = {}  # Dictionary to store delay and energy consumption for each user

        for user in self.users:
            chosen_model_m = model_selection_dic[user]["model"]  # Get the selected model for this user
            bandwidth_m = bandwidth_allocation_dic[user]
            data_size_m = self.data_size[chosen_model_m]
            coded_data_size_m = data_size_m / ldpc_rate_dic[user]
            trans_delay = coded_data_size_m / (bandwidth_m * trans_rate_dic[user])
            trans_energy = self.md_params[user]['trans_power'] * trans_delay

            # Store delay and energy in the result dictionary for the user
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

            # Compute local processing delay
            edge_delay = tail_flops_m * 1e-9 / (gpu_freq_m * num_cores_m * flops_per_cycle_m)

            # Compute local energy consumption
            edge_energy = self.md_params[user]['power_coeff'] * gpu_freq_m ** 3 * edge_delay

            # Store delay and energy in the result dictionary for the user
            edge_overhead_dic[user] = {
                "delay": edge_delay,
                "energy": edge_energy
            }

        return edge_overhead_dic  # Return dictionary with local overhead for all users

    def get_total_overhead(self, local_overhead_dic, trans_overhead_dic, edge_overhead_dic):
        total_overhead_dic = {}  # Dictionary to store delay and energy consumption for each user
        # print("local overhead", local_overhead_dic)
        # print("trans overhead", trans_overhead_dic)
        # print("edge overhead", edge_overhead_dic)
        for user in self.users:
            total_delay = local_overhead_dic[user]['delay'] + trans_overhead_dic[user]['delay'] + \
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

        metrics = ["delay", "energy", "accuracy", "reward", "is_vio", "vio_degree"]
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
        }

        # Compute standard deviations
        self.std_metrics = {
            "latency": float(np.std(metric_values["delay"], ddof=1)),
            "energy": float(np.std(metric_values["energy"], ddof=1)),
            "accuracy": float(np.std(metric_values["accuracy"], ddof=1)),
            "reward": float(np.std(metric_values["reward"], ddof=1)),
            "vio_prob": float(np.std(metric_values["is_vio"], ddof=1)),
            "vio_sum": float(np.std(metric_values["vio_degree"], ddof=1)),
        }

        # Normalize action frequency
        self.action_freq = self.action_freq / self.time_slot_num

    def get_instant_metrics(self, task_dic, total_overhead_dic, reward_dic, acc_dic):
        for user in self.users:
            self.instant_metrics[user]["delay"].append(total_overhead_dic[user]['delay'])
            self.instant_metrics[user]["energy"].append(total_overhead_dic[user]['energy'])
            self.instant_metrics[user]["accuracy"].append(acc_dic[user])
            self.instant_metrics[user]["reward"].append(reward_dic[user])  # Store reward for the current time slot
            reward_dic[user] = float(reward_dic[user])

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
            # Estimate the achievable rate
            snr_dic, trans_rate_dic = self.get_trans_rate(t)

            # Task generation
            task_dic = self.generate_tasks(t)

            # MDs observe the context and select ML models based on GP
            context_dic = self.observe_context(task_dic, trans_rate_dic)
            model_selection_dic = self.model_selection(context_dic)

            # Calculate the local processing overhead
            local_overhead_dic = self.get_local_overhead(model_selection_dic)

            # The ES performs BCD-based optimization
            # Initialize variables for the BCD (Block Coordinate Descent) algorithm

            bcd_obj_last = float('inf')  # Previous objective function value (used for convergence check)
            bcd_iter = 1  # Iteration counter

            while True:
                # Choose initialization or update step based on the current iteration
                if bcd_iter == 1:
                    # Initialization: Assign initial LDPC rates for each user randomly
                    init_ldpc_rate_dic = {user: random.choice(self.available_coding_rate) for user in self.users}
                    # Allocate bandwidth using the initial LDPC rate dictionary
                    bandwidth_allocation_dic = self.allocate_bandwidth(task_dic, model_selection_dic, trans_rate_dic,
                                                                       init_ldpc_rate_dic)
                else:
                    # Update: Allocate bandwidth using the current LDPC rate dictionary
                    bandwidth_allocation_dic = self.allocate_bandwidth(task_dic, model_selection_dic, trans_rate_dic,
                                                                       ldpc_rate_dic)
                # Allocate GPU resources for the users
                gpu_allocation_dic = self.gpu_resource_allocation(task_dic, model_selection_dic)

                # Select the coding rate for each user based on current data
                ldpc_rate_dic = self.coding_rate_selection(task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                                                           local_overhead_dic, bandwidth_allocation_dic,
                                                           gpu_allocation_dic)

                # Get the accuracy for each user based on current SNR, coding rate, and model selection
                acc_dic = self.get_accuracy(snr_dic, ldpc_rate_dic, model_selection_dic)

                # Get the transmission overhead for each user
                trans_overhead_dic = self.get_trans_overhead(trans_rate_dic, model_selection_dic,
                                                             bandwidth_allocation_dic, ldpc_rate_dic)

                # Get the edge processing overhead for each user
                edge_overhead_dic = self.get_edge_overhead(model_selection_dic, gpu_allocation_dic)

                # Calculate the total overhead for each user (local + transmission + edge processing)
                total_overhead_dic = self.get_total_overhead(local_overhead_dic, trans_overhead_dic, edge_overhead_dic)
                # Find the user with the minimum accuracy and the corresponding accuracy value
                bcd_min_acc_user = min(acc_dic, key=lambda user: acc_dic[user].item())
                bcd_min_acc_value = acc_dic[bcd_min_acc_user].item()

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

            # Calculate the reward for MDs and update the GP
            reward_dic = self.get_reward(task_dic, acc_dic, total_overhead_dic)
            self.get_instant_metrics(task_dic, total_overhead_dic, reward_dic, acc_dic)
            self.update_gp(context_dic, model_selection_dic, reward_dic)

        self.get_average_and_std_metrics()


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

