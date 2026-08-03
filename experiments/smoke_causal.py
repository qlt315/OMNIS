"""Quick smoke test: compare causal MAB vs conference GP-UCB on a small config."""
import time
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sys_data.config import Config
from omnis.omnis_main import OMNIS


def run(algo, seed, user_num=3, slots=60):
    config = Config(seed)
    config.update_users(user_num)
    config.time_slot_num = slots
    config.algo = algo
    omnis = OMNIS(config)
    t0 = time.time()
    omnis.simulation()
    elapsed = time.time() - t0
    return omnis, elapsed


if __name__ == "__main__":
    for algo in ["ucb", "causal"]:
        omnis, elapsed = run(algo, seed=0)
        print(f"\n=== {algo.upper()} ===  ({elapsed:.1f}s)")
        print("aver info:", {k: round(v, 4) for k, v in omnis.average_metrics.items()})
        print("action freq:", np.round(omnis.action_freq, 3))
