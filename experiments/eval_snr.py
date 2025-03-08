import numpy as np
import scipy.io as sio
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.cto_main import CTO
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.rss_main import RSS
from sys_data.trans_sys_sim.mimo_channel_gen.mimo_channel_gen_fix_snr import snr_values

# Initialize parameters
snr_list = snr_values
snr_num = len(snr_list)
seed = 42

# Define metric names
metric_names = ["reward", "latency", "energy", "accuracy", "vio_prob", "vio_sum"]
num_algorithms = 5

# Initialize storage for evaluation results
eval_results = {name: np.zeros([num_algorithms, snr_num]) for name in metric_names}

# Define baseline algorithms
algorithms = [OMNIS, CTO, DTS, GDO, RSS]
algorithm_names = ["OMNIS", "CTO", "DTS", "GDO", "RSS"]

for snr_idx, snr in enumerate(snr_list):
    print(f"Evaluating SNR (dB): {snr}")

    # Load system configuration and channel data
    config = Config(seed)
    channel_data_filename = f"sys_data/trans_sys_sim/mimo_channel_gen/mimo_channel_data_snr_{snr}.npy"
    config.channel_data = np.load(channel_data_filename)

    # Evaluate each algorithm
    for alg_idx, (alg_cls, alg_name) in enumerate(zip(algorithms, algorithm_names)):
        print(f"Evaluating {alg_name}...")
        algorithm = alg_cls(config)
        algorithm.simulation()

        # Store results
        for metric in metric_names:
            eval_results[metric][alg_idx, snr_idx] = algorithm.average_metrics[metric]

# Save results to .mat file
sio.savemat("experiments/eval_diff_snr.mat", eval_results)
print("Data successfully saved to experiments/eval_diff_snr.mat")