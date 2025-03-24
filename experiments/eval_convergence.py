import numpy as np
import scipy.io as sio
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS

# Define the random seed list, number of users, and number of time slots
seed_list = [0,37,42]
seed_num = len(seed_list)
config = Config(0)  # just roughly initialize to get the  number of MD and time slots.
# Metrics to be recorded
metrics = ["reward", "delay", "accuracy", "energy", "is_vio", "vio_degree"]
algorithms = {
    "omnis": OMNIS,
    "cto": CTO,
    "dts": DTS,
    "gdo": GDO,
    "rss": RSS
}

# Initialize dictionaries to store results for each algorithm and metric
results = {f"{algo}_ins_{metric}": np.zeros((seed_num, config.user_num, config.time_slot_num)) for algo in algorithms for metric in metrics}
aver_rewards = {f"{algo}_ins_aver_reward": np.zeros((config.user_num, config.time_slot_num)) for algo in algorithms}

# Iterate over different random seeds
for seed_idx, seed in enumerate(seed_list):
    print(f"Evaluating Seed: {seed}")

    # Run simulations for each algorithm, reinitialize config each time
    for name, cls in algorithms.items():
        print(f"Evaluating {name.upper()}...")

        # Reinitialize config for each algorithm evaluation
        alg_config = Config(seed)
        alg_config.user_num, alg_config.time_slot_num = config.user_num, config.time_slot_num
        alg_config.users = [f"user_{i + 1}" for i in range(config.user_num)]
        alg_config.fixed_delay = {user: np.random.uniform(0.5, 1) for user in alg_config.users}
        alg_config.fixed_energy = {user: np.random.uniform(0.1, 1.0) for user in alg_config.users}
        alg_config.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in alg_config.users}

        # Initialize the algorithm instance with the updated config
        instance = cls(alg_config)
        instance.simulation()

        # Store instant metrics for each user and time slot
        for t in range(config.time_slot_num):
            for user_idx, user in enumerate(alg_config.users):
                for metric in metrics:
                    results[f"{name}_ins_{metric}"][seed_idx, user_idx, t] = instance.instant_metrics[user][metric][t]

# After collecting all data for all seeds, compute the average across seeds for each algorithm
for name in algorithms:
    for metric in metrics:
        # Compute the average metric over all seeds, for all users, and across time slots
        metric_data = results[f"{name}_ins_{metric}"]  # (seed_num, user_num, time_slot_num)
        averaged_metric = np.mean(metric_data, axis=0)  # Average over seeds
        results[f"{name}_ins_{metric}"] = averaged_metric  # Store averaged metric

    # Compute the average reward over all users for each time slot
    aver_rewards[f"{name}_ins_reward"] = np.mean(np.sum(results[f"{name}_ins_reward"], axis=1), axis=0)

# Only keep the seed-averaged results (user_num x time_slot_num matrices)
averaged_results = {key: results[key] for key in results}

# Save the averaged results to a .mat file (no seed-specific data)
sio.savemat("experiments/results/eval_convergence.mat", averaged_results)

print("Seed-averaged data successfully saved to experiments/results/eval_convergence.mat")
