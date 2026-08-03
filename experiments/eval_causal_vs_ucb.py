"""Journal experiment: OMNIS-Causal vs OMNIS-UCB (conference version).

Runs both algorithms on the default configuration over multiple seeds and
saves per-slot convergence traces plus summary metrics to
experiments/results/eval_causal_vs_ucb.npz
"""
import time
import numpy as np
import scipy.io as sio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sys_data.config import Config
from omnis.omnis_main import OMNIS

SEED_LIST = [0, 37, 42]
ALGO_LIST = ["ucb", "causal"]
METRICS = ["reward", "latency", "energy", "accuracy", "vio_prob", "vio_sum"]


def run(algo, seed):
    config = Config(seed)
    config.algo = algo
    omnis = OMNIS(config)
    t0 = time.time()
    omnis.simulation()
    elapsed = time.time() - t0
    return omnis, elapsed


if __name__ == "__main__":
    num_algos = len(ALGO_LIST)
    avg = {m: np.zeros(num_algos) for m in METRICS}
    traces = {m: [] for m in ["reward", "accuracy", "delay", "vio_prob"]}
    decision_time = np.zeros(num_algos)
    update_time = np.zeros(num_algos)
    total_time = np.zeros(num_algos)

    for a_idx, algo in enumerate(ALGO_LIST):
        trace_acc = []
        for seed in SEED_LIST:
            print(f"[{algo}] seed={seed} ...", flush=True)
            omnis, elapsed = run(algo, seed)
            print(f"    aver: { {k: round(v, 4) for k, v in omnis.average_metrics.items()} }  "
                  f"(total {elapsed:.1f}s, decision {omnis.decision_time:.1f}s, update {omnis.update_time:.1f}s)",
                  flush=True)
            for m in METRICS:
                avg[m][a_idx] += omnis.average_metrics[m] / len(SEED_LIST)
            decision_time[a_idx] += omnis.decision_time / len(SEED_LIST)
            update_time[a_idx] += omnis.update_time / len(SEED_LIST)
            total_time[a_idx] += elapsed / len(SEED_LIST)
            trace_acc.append(np.mean([omnis.instant_metrics[u]["reward"] for u in omnis.users], axis=0))
        traces["reward"].append(np.mean(trace_acc, axis=0))

    print("\n===== Seed-averaged results =====")
    header = f"{'algo':<8}" + "".join(f"{m:>10}" for m in METRICS) + f"{'dec_t':>8}{'upd_t':>8}{'tot_t':>8}"
    print(header)
    for a_idx, algo in enumerate(ALGO_LIST):
        row = f"{algo:<8}" + "".join(f"{avg[m][a_idx]:>10.4f}" for m in METRICS)
        row += f"{decision_time[a_idx]:>8.2f}{update_time[a_idx]:>8.2f}{total_time[a_idx]:>8.2f}"
        print(row)

    out = {f"{m}": avg[m] for m in METRICS}
    out.update({
        "algo_names": np.array(ALGO_LIST),
        "decision_time": decision_time,
        "update_time": update_time,
        "total_time": total_time,
        "reward_trace": np.array(traces["reward"]),
    })
    os.makedirs("experiments/results", exist_ok=True)
    np.savez("experiments/results/eval_causal_vs_ucb.npz", **out)
    sio.savemat("experiments/results/eval_causal_vs_ucb.mat", out)
    print("\nSaved to experiments/results/eval_causal_vs_ucb.npz / .mat")
