import numpy as np
import time
import scipy.io as sio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS
from sys_data.mimo_channel_gen.mimo_channel_gen_fix_snr import snr_values

# Initialize parameters
snr_list = snr_values
snr_num = len(snr_list)
seed_list = [0, 37, 42]  # Define the list of seeds to average over
seed_num = len(seed_list)

# Define metric names
metric_names = ["reward", "latency", "energy", "accuracy", "vio_prob", "vio_sum"]
metric_save_names = ["reward_diff_snr", "latency_diff_snr", "energy_diff_snr",
                     "accuracy_diff_snr", "vio_prob_diff_snr", "vio_sum_diff_snr"]
num_algorithms = 5

# Initialize storage for evaluation results (for each metric)
eval_results = {name: np.zeros([num_algorithms, snr_num]) for name in metric_save_names}

# Define baseline algorithms
algorithms = [OMNIS, CTO, DTS, GDO, RSS]
algorithm_names = ["OMNIS", "CTO", "DTS", "GDO", "RSS"]

# Iterate over each seed and perform the simulations
for seed_idx, seed in enumerate(seed_list):
    print(f"Evaluating Seed: {seed}")

    # Iterate over each SNR value
    for snr_idx, snr in enumerate(snr_list):
        print(f"Evaluating SNR (dB): {snr}")

        # Iterate over each algorithm
        for alg_idx, (alg_cls, alg_name) in enumerate(zip(algorithms, algorithm_names)):
            print(f"Evaluating {alg_name} at SNR {snr} dB with seed = {seed}...")

            # Reinitialize Config for each algorithm to avoid shared state issues
            config = Config(seed)
            channel_data_filename = f"sys_data/mimo_channel_gen/mimo_channel_data_snr_{snr}.npy"
            config.channel_data = np.load(channel_data_filename)

            # Initialize and run the algorithm
            algorithm = alg_cls(config)
            algorithm.simulation()
            print("aver info:", algorithm.average_metrics)
            # print("std info:", algorithm.std_metrics)
            # print("action freq info:", algorithm.action_freq)

            # Store the results for each metric and each algorithm, averaging over all seeds
            for metric, save_metric in zip(metric_names, metric_save_names):
                eval_results[save_metric][alg_idx, snr_idx] += algorithm.average_metrics[metric]

# After collecting results for all seeds, compute the average across all seeds for each algorithm and SNR
for save_metric in eval_results:
    eval_results[save_metric] /= seed_num  # Average over the seeds

# Save the averaged results to a .mat file
sio.savemat("experiments/results/eval_diff_snr.mat", eval_results)
print("Seed-averaged data successfully saved to experiments/results/eval_diff_snr.mat")
