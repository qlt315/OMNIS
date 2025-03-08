import numpy as np
import scipy.io as sio
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS

# Define different user numbers for evaluation
user_num_list = [1, 2, 3, 4, 5, 6, 7, 8]
num_user_cases = len(user_num_list)
seed = 42

# Define metric names
metric_names = ["reward", "latency", "energy", "accuracy", "vio_prob", "vio_sum"]
num_algorithms = 5

# Initialize storage for evaluation results
eval_results = {name: np.zeros([num_algorithms, num_user_cases]) for name in metric_names}

# Define baseline algorithms
algorithms = [OMNIS, CTO, DTS, GDO, RSS]
algorithm_names = ["OMNIS", "CTO", "DTS", "GDO", "RSS"]

# Iterate over different numbers of users
for user_idx, user_num in enumerate(user_num_list):
    print(f"Evaluating User Number: {user_num}")

    # Load system configuration
    config = Config(seed)
    config.update_users(user_num)

    # Evaluate each algorithm
    for alg_idx, (alg_cls, alg_name) in enumerate(zip(algorithms, algorithm_names)):
        print(f"Evaluating {alg_name}...")
        algorithm = alg_cls(config)
        algorithm.simulation()

        # Store results
        for metric in metric_names:
            eval_results[metric][alg_idx, user_idx] = algorithm.average_metrics[metric]

# Save results to .mat file
sio.savemat("experiments/eval_diff_users.mat", eval_results)
print("Data successfully saved to experiments/eval_diff_users.mat")
