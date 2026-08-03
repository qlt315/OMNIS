"""Journal ablation: isolate the contribution of each causal-MAB component.

Variants:
    ucb            conference-version GP-UCB (6-dim reward GP per MD)
    causal         full causal MAB (prior + shared mechanism GP + ES marginalization)
    causal_noprior causal MAB without the offline prior mean
    causal_noshare causal MAB without cross-MD mechanism pooling
    causal_ts      causal MAB with Thompson sampling acquisition
"""
import time
import numpy as np
import scipy.io as sio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sys_data.config import Config
from omnis.omnis_main import OMNIS

SEED_LIST = [0, 37, 42]
VARIANTS = {
    "ucb": {"algo": "ucb"},
    "causal": {"algo": "causal"},
    "causal_noprior": {"algo": "causal", "causal_use_prior": False},
    "causal_noshare": {"algo": "causal", "causal_shared": False},
    "causal_ts": {"algo": "causal", "causal_acq": "ts"},
}
METRICS = ["reward", "latency", "energy", "accuracy", "vio_prob", "vio_sum"]


def run(overrides, seed):
    config = Config(seed)
    for key, value in overrides.items():
        setattr(config, key, value)
    omnis = OMNIS(config)
    t0 = time.time()
    omnis.simulation()
    elapsed = time.time() - t0
    return omnis, elapsed


if __name__ == "__main__":
    names = list(VARIANTS.keys())
    avg = {m: np.zeros(len(names)) for m in METRICS}
    decision_time = np.zeros(len(names))
    update_time = np.zeros(len(names))
    total_time = np.zeros(len(names))

    for v_idx, name in enumerate(names):
        for seed in SEED_LIST:
            print(f"[{name}] seed={seed} ...", flush=True)
            omnis, elapsed = run(VARIANTS[name], seed)
            print(f"    aver: { {k: round(v, 4) for k, v in omnis.average_metrics.items()} }  "
                  f"(total {elapsed:.1f}s, decision {omnis.decision_time:.1f}s, update {omnis.update_time:.1f}s)",
                  flush=True)
            for m in METRICS:
                avg[m][v_idx] += omnis.average_metrics[m] / len(SEED_LIST)
            decision_time[v_idx] += omnis.decision_time / len(SEED_LIST)
            update_time[v_idx] += omnis.update_time / len(SEED_LIST)
            total_time[v_idx] += elapsed / len(SEED_LIST)

    print("\n===== Ablation (seed-averaged) =====")
    header = f"{'variant':<15}" + "".join(f"{m:>10}" for m in METRICS) + f"{'dec_t':>8}{'upd_t':>8}{'tot_t':>8}"
    print(header)
    for v_idx, name in enumerate(names):
        row = f"{name:<15}" + "".join(f"{avg[m][v_idx]:>10.4f}" for m in METRICS)
        row += f"{decision_time[v_idx]:>8.2f}{update_time[v_idx]:>8.2f}{total_time[v_idx]:>8.2f}"
        print(row)

    out = {f"{m}": avg[m] for m in METRICS}
    out.update({
        "variant_names": np.array(names),
        "decision_time": decision_time,
        "update_time": update_time,
        "total_time": total_time,
    })
    os.makedirs("experiments/results", exist_ok=True)
    np.savez("experiments/results/eval_causal_ablation.npz", **out)
    sio.savemat("experiments/results/eval_causal_ablation.mat", out)
    print("\nSaved to experiments/results/eval_causal_ablation.npz / .mat")
