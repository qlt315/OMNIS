import numpy as np
import scipy.io as sio
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS

# Define different user numbers for evaluation
user_num_list = [2, 3, 4, 5, 6, 7, 8, 9]
num_user_cases = len(user_num_list)
seed_list = [0, 37, 42]  # Define the list of seeds to average over
seed_num = len(seed_list)

# Define metric names
metric_names = ["reward", "latency", "energy", "accuracy", "vio_prob", "vio_sum"]
metric_save_names = ["reward_diff_user_num", "latency_diff_user_num", "energy_diff_user_num",
                     "accuracy_diff_user_num", "vio_prob_diff_user_num", "vio_sum_diff_user_num"]
num_algorithms = 5

# Initialize storage for evaluation results (for each metric)
eval_results = {name: np.zeros([num_algorithms, num_user_cases]) for name in metric_save_names}

# Define baseline algorithms
algorithms = [OMNIS, CTO, DTS, GDO, RSS]
algorithm_names = ["OMNIS", "CTO", "DTS", "GDO", "RSS"]

# Iterate over different seeds
for seed_idx, seed in enumerate(seed_list):
    print(f"Evaluating Seed: {seed}")

    # Iterate over different numbers of users
    for user_idx, user_num in enumerate(user_num_list):
        print(f"Evaluating User Number: {user_num}")

        # Iterate over each algorithm
        for alg_idx, (alg_cls, alg_name) in enumerate(zip(algorithms, algorithm_names)):
            print(f"Evaluating {alg_name} with {user_num} users with seed = {seed}...")

            # Reinitialize config for each algorithm to avoid shared state
            config = Config(seed)
            config.update_users(user_num)
            print("fix delay:",config.fixed_delay)

            # Initialize and run the algorithm
            algorithm = alg_cls(config)
            algorithm.simulation()
            print("aver info:", algorithm.average_metrics)
            # print("std info:", algorithm.std_metrics)
            # print("action freq info:", algorithm.action_freq)
            # Store the results for each metric and each algorithm, averaging over all seeds
            for metric, save_metric in zip(metric_names, metric_save_names):
                eval_results[save_metric][alg_idx, user_idx] += algorithm.average_metrics[metric]

# After collecting results for all seeds, compute the average across all seeds for each algorithm and user number
for metric in eval_results:
    eval_results[metric] /= seed_num  # Average over the seeds

# Save the averaged results to a .mat file
sio.savemat("experiments/results/eval_diff_user_num.mat", eval_results)
print("Seed-averaged data successfully saved to experiments/results/eval_diff_user_num.mat")
