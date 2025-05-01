import os
import sys
import numpy as np
import scipy.io as sio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS

# Set parameters
snr_list = [2, 4, 6]  # SNR values to evaluate
user_num_list = [2, 4, 6]  # Number of users to evaluate
seed_list = [0, 37, 42]  # List of seeds to average over
seed_num = len(seed_list)

# Define algorithms
algorithms = [OMNIS, CTO, DTS, GDO, RSS]
algorithm_names = ["OMNIS", "CTO", "DTS", "GDO", "RSS"]

# Initialize the configuration with the first seed
config = Config(seed_list[0])
action_num = len(config.models)

# Initialize 3D arrays to store the average action frequencies for SNR and user number variations
action_freq_snr = np.zeros((len(snr_list), len(algorithms), action_num))  # [SNR, algorithms, action_num]
action_freq_users = np.zeros((len(user_num_list), len(algorithms), action_num))  # [user_num, algorithms, action_num]

# Iterate over different SNR values
for snr_idx, snr in enumerate(snr_list):
    print(f"Evaluating SNR: {snr} dB with 5 users")

    # Run each algorithm and store the average action frequency
    for alg_idx, (alg_cls, alg_name) in enumerate(zip(algorithms, algorithm_names)):
        # Load system configuration for each seed
        avg_action_freq = np.zeros(action_num)  # Initialize an array to accumulate action frequencies for each seed
        for seed_idx, seed in enumerate(seed_list):
            config = Config(seed)
            channel_data_filename = f"sys_data/mimo_channel_gen/mimo_channel_data_snr_{snr}.npy"
            config.channel_data = np.load(channel_data_filename)

            print(f"Running {alg_name} with seed {seed}...")
            algorithm = alg_cls(config)
            algorithm.simulation()

            # Calculate the average action frequency across users (mean along user dimension)
            avg_action_freq += np.mean(algorithm.action_freq, axis=0)  # Result: [action_num]

        # Average over all seeds
        action_freq_snr[snr_idx, alg_idx, :] = avg_action_freq / seed_num  # Store the result in the array

# Iterate over different user numbers
for user_num_idx, user_num in enumerate(user_num_list):
    print(f"Evaluating user_num: {user_num} with default channel data")

    # Run each algorithm and store the average action frequency
    for alg_idx, (alg_cls, alg_name) in enumerate(zip(algorithms, algorithm_names)):
        # Load system configuration for each seed
        avg_action_freq = np.zeros(action_num)  # Initialize an array to accumulate action frequencies for each seed
        for seed_idx, seed in enumerate(seed_list):
            config = Config(seed)
            config.update_users(user_num)  # Update the number of users

            print(f"Running {alg_name} with seed {seed}...")
            algorithm = alg_cls(config)
            algorithm.simulation()

            # Calculate the average action frequency across users (mean along user dimension)
            avg_action_freq += np.mean(algorithm.action_freq, axis=0)  # Result: [action_num]

        # Average over all seeds
        action_freq_users[user_num_idx, alg_idx, :] = avg_action_freq / seed_num  # Store the result in the array

# Save the results
sio.savemat("experiments/results/action_freq_snr.mat", {"action_freq_snr": action_freq_snr})
sio.savemat("experiments/results/action_freq_user_num.mat", {"action_freq_user_num": action_freq_users})

print("Seed-averaged action frequency data saved successfully for SNR and user number variations.")
