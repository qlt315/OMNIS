import numpy as np
import scipy.io as sio
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS

# Define the random seed list, number of users, and number of time slots
seed_list = [0, 37, 42]
seed_num, user_num, time_slot_num = len(seed_list), 3, 150

# Metrics to be recorded
metrics = ["reward", "delay", "accuracy", "energy"]
algorithms = {
    "omnis": OMNIS,
    "cto": CTO,
    "dts": DTS,
    "gdo": GDO,
    "rss": RSS
}

# Initialize dictionaries to store results for each algorithm and metric
results = {f"{algo}_ins_{metric}": np.zeros((seed_num, user_num, time_slot_num)) for algo in algorithms for metric in metrics}
aver_rewards = {f"{algo}_ins_aver_reward": np.zeros((seed_num, time_slot_num)) for algo in algorithms}

# Iterate over different random seeds
for seed_idx, seed in enumerate(seed_list):
    print(f"Evaluating Seed: {seed}")
    config = Config(seed)
    config.user_num, config.time_slot_num = user_num, time_slot_num
    config.users = [f"user_{i + 1}" for i in range(user_num)]
    config.fixed_delay = {user: np.random.uniform(0.5, 1) for user in config.users}  # Delay constraints
    config.fixed_energy = {user: np.random.uniform(0.1, 1.0) for user in config.users}  # Energy constraints
    config.fixed_energy_weight = {user: np.random.uniform(0.3, 0.7) for user in config.users}  # Energy weight factors

    # Initialize all algorithm instances
    instances = {name: cls(config) for name, cls in algorithms.items()}

    # Run simulations for all algorithms
    for name, instance in instances.items():
        print(f"Evaluating {name.upper()}...")
        instance.simulation()

    # Store instant metrics for each user and time slot
    for t in range(time_slot_num):
        for user_idx, user in enumerate(config.users):
            for name, instance in instances.items():
                for metric in metrics:
                    results[f"{name}_ins_{metric}"][seed_idx, user_idx, t] = instance.instant_metrics[user][metric][t]

    # Compute the average reward over all users for each time slot
    for name in algorithms:
        aver_rewards[f"{name}_ins_aver_reward"] = np.mean(np.sum(results[f"{name}_ins_reward"], axis=1), axis=0)

# Save results to a .mat file
save_dict = {**results, **aver_rewards}
sio.savemat("experiments/eval_convergence.mat", save_dict)

print("Data successfully saved to experiments/eval_convergence.mat")
