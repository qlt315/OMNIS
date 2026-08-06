# OMNIS

## Overview

Edge computing enables resource-constrained devices to execute machine learning applications via task offloading. To this aim, radio access network (RAN) slicing is instrumental to provide the necessary communication and computing resources. However, current RAN slicing approaches rely on static computing models, thereby constraining their ability to leverage the dynamic semantic data representation capabilities enabled by recent neural architectures. In this paper, we propose OMNIS, a semantic RAN slicing framework for edge computing built on a new generation of dynamic split neural models. In contrast to prior work, OMNIS embeds a dynamic form of neural compression paired with adaptive data encoding for task offloading, which provides an ample set of communication payload and computing options for RAN slicing. Differently from prior methodologies for semantic communications, we explicitly study the interplay between neural compression and information quantization in determining the performance of computer vision tasks. In this context, we design a new quantization approach, named ``box'' quantization, which improves resiliency to bit errors as a function of the compression rate compared to current state of the art. Considering the partial observability and differing objectives of the network nodes, we formulate two interdependent optimization problems to achieve optimal inference performance: (i) the mobile devices (MDs) maximize inference accuracy under quality of service (QoS) constraints by controlling the dynamic split deep neural networks (DNNs), and (ii) the edge server (ES) allocates bandwidth and computing resources to maximize the worst inference accuracy among all MDs. To solve these problems, we propose a multi-agent distributed optimization framework, where MDs act as contextual multi-armed bandit (MAB) agents using Bayesian optimization, and the ES performs resource allocation via convex optimization. Compared to existing RAN slicing schemes for edge computing, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x. Compared to existing RAN slicing frameworks, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x.

Journal extension: multi-cell Top-L association, queueing + Lyapunov control, causal MAB, Sionna PHY tables/traces, RL baselines. Reported **reward** is the Lyapunov objective \(V\cdot u+\mathrm{drift}\) (not the QoS utility alone). Plots/CSVs: `figures/`.

## Structure
```bash
OMNIS/
│── baselines/                      # RSS, DTS, CTO, GDO, DQN, PPO, MAPPO
│── omnis/                          # Simulator, causal MAB, PHY helpers
│── experiments/
│   ├── train_lib.py                # Shared runner (data only)
│   ├── train_all.py                # Selected / all schemes → figures/
│   ├── train_*.py                  # Per-scheme trainers
│   ├── comm_model.py               # Control-plane bytes / RTT → comm_ms
│   └── plot_results.py             # Plot from saved CSVs + series/
│── phy_sim/                        # Sionna PHY; see phy_sim/README.md
│── sys_data/
│   ├── config.py                   # Config
│   └── acc_data/                   # Optional seed curves for phy_sim/synth_acc
│── figures/                        # train output + plots
│── observations/                   # Observation figure scripts
```

## How to Try OMNIS

Set `PYTHONPATH=.` and run from the repository root.

### 1. PHY tables / SINR traces
```bash
# See phy_sim/README.md
python3 phy_sim/run_bler.py ...
python3 phy_sim/run_acc.py ...
python3 phy_sim/run_traces.py ...
```

### 2. Train one scheme
```bash
PYTHONPATH=. python3 experiments/train_causal.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures
PYTHONPATH=. python3 experiments/train_dqn.py --slots 80 --seeds 0 --out figures
# same flags for: train_ucb / train_dts / train_gdo / train_rss / train_ppo / train_mappo / train_cto
```

### 3. Train selected / all schemes
```bash
# all schemes
PYTHONPATH=. python3 experiments/train_all.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures
# subset (merges into existing perseed.csv; does not wipe other algos)
PYTHONPATH=. python3 experiments/train_all.py --algos causal ucb gdo dqn ppo mappo --out figures
PYTHONPATH=. python3 experiments/train_all.py --algos cto --out figures
```

### 4. Plot (separate; skips missing algos)
```bash
PYTHONPATH=. python3 experiments/plot_results.py --indir figures
PYTHONPATH=. python3 experiments/plot_results.py --indir figures --algos causal ucb cto
```

Outputs under `--out` / `--indir` (default `figures/`):
- `perseed.csv` / `summary.csv` — reward, acc, delay, energy, backlog, vio, ms/slot
  (incl. `decision_ms`, `comm_ms`, `bcd_ms`; new runs fold update into `decision_ms`)
- `series/{name}_seed{k}.npz` — time series for curves
- plots from `plot_results.py`:
  - series: `reward.png`, `reward_sliding.png`, `accuracy.png`, `delay.png`,
    `energy.png`, `backlog.png`, `violation.png`
  - mean bars (across seeds, ± std): `reward_bar.png`, `accuracy_bar.png`,
    `delay_bar.png`, `energy_bar.png`, `backlog_bar.png`, `violation_bar.png`
  - one stacked `runtime.png` (**decision** = algo compute, **interaction** =
    control-plane RTT+bytes via `comm_model.py`, **BCD** = resource allocation).
    Old CSVs without `comm_*` derive interaction at plot time. Optional `--users`
    overrides Config.user_num for that derivation.
- `plot_data.mat` — MATLAB export (`algo_names`, `summary.*`, `series.<algo>.*`,
  plus flat `decision_ms` / `comm_ms` / `bcd_ms` / …)

## Algorithms

| Name | Role |
|------|------|
| **causal** | OMNIS-Causal |
| **ucb** | OMNIS-UCB |
| **dts** | OMNIS-TS |
| **cto** | Centralized joint GP-UCB |
| **gdo** | SF-ESP greedy |
| **rss** | Static random arms |
| **dqn** | Centralized **joint-action** Double-DQN (candidate pool like CTO) |
| **ppo** | Centralized branching PPO |
| **mappo** | Multi-agent PPO (CTDE) |

## Important Notes
1. Prefer running with the working directory set to the repo root and `PYTHONPATH=.`.
2. Ensure SINR traces cover at least `config.time_slot_num` slots and enough UEs for `config.user_num`.
3. Hyperparameters live in `sys_data/config.py`. Fair Causal uses `causal_drift_gain=1.0`
   (same `V·u+drift` objective as other schemes); shared knobs (`lyapunov_v`,
   `reward_w_acc`) and Causal method strengths (`causal_beta`, GP length scales /
   prior) may be retuned — do not reintroduce a Causal-only soft-queue gain.
   CTO intentionally retains full joint CBO cost: `cto_gp_burn_in=0` re-fits
   ARD hypers every slot (`cto_gp_n_restarts` multi-start L-BFGS),
   `cto_max_candidates` (default 46656 ≈ 6^6) scores a large on-the-fly joint
   pool with stock sklearn GP predict (`cto_use_fast_gp=False`) — never
   materializing `(n_models·L)^U` (avoids OOM). FastGP remains for per-user
   Causal/UCB. Positive `cto_gp_burn_in` freezes hypers after N (optional).
4. This repository does not include training code for the multi-branch dynamic split DNN; it interfaces to evaluation tables. For DNN details, contact Ian Andrew Harshbarger (iharshba@uci.edu).
5. Runtime plots report one stacked bar: **decision** (algo compute) + **interaction** (control-plane) + **BCD**. Metric mean bars are separate `*_bar.png` files.

## Contributing
We welcome contributions to improve OMNIS. To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork.
4. Open a pull request detailing the changes.

---

## Acknowledgments

The multi-agent MAB optimization scheme is developed based on https://github.com/jaayala/contextual_bayesian_optimization
