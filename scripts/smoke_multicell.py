"""Smoke test: multi-cell joint (model, cell_rank) arms with SINR traces."""
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sys_data.config import Config
from omnis.omnis_main import OMNIS


def run(algo, seed, user_num=6, slots=20):
    config = Config(seed)
    config.update_users(user_num)
    config.time_slot_num = slots
    config.algo = algo
    omnis = OMNIS(config)
    t0 = time.time()
    omnis.simulation()
    elapsed = time.time() - t0
    return omnis, elapsed


def cell_histogram(omnis):
    counts = Counter()
    for user in omnis.users:
        for cell in omnis.instant_metrics[user]["cell"]:
            counts[cell] += 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    for algo in ["causal", "ucb"]:
        omnis, elapsed = run(algo, seed=0)
        print(f"\n=== {algo.upper()} ({elapsed:.1f}s) ===")
        print("avg metrics:", {k: round(v, 4) for k, v in omnis.average_metrics.items()
                               if k in ("reward", "accuracy", "backlog_bits", "latency", "energy")})
        print("cell histogram:", cell_histogram(omnis))
        print("action freq (model x cell_rank) sum:", np.round(omnis.action_freq.sum(axis=0), 3))
