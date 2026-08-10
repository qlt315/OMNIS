# OMNIS

## Overview

Edge computing enables resource-constrained devices to execute machine learning applications via task offloading. To this aim, radio access network (RAN) slicing is instrumental to provide the necessary communication and computing resources. However, current RAN slicing approaches rely on static computing models, thereby constraining their ability to leverage the dynamic semantic data representation capabilities enabled by recent neural architectures. In this paper, we propose OMNIS, a semantic RAN slicing framework for edge computing built on a new generation of dynamic split neural models. In contrast to prior work, OMNIS embeds a dynamic form of neural compression paired with adaptive data encoding for task offloading, which provides an ample set of communication payload and computing options for RAN slicing. Differently from prior methodologies for semantic communications, we explicitly study the interplay between neural compression and information quantization in determining the performance of computer vision tasks. In this context, we design a new quantization approach, named ``box'' quantization, which improves resiliency to bit errors as a function of the compression rate compared to current state of the art. Considering the partial observability and differing objectives of the network nodes, we formulate two interdependent optimization problems to achieve optimal inference performance: (i) the mobile devices (MDs) maximize inference accuracy under quality of service (QoS) constraints by controlling the dynamic split deep neural networks (DNNs), and (ii) the edge server (ES) allocates bandwidth and computing resources to maximize the worst inference accuracy among all MDs. To solve these problems, we propose a multi-agent distributed optimization framework, where MDs act as contextual multi-armed bandit (MAB) agents using Bayesian optimization, and the ES performs resource allocation via convex optimization. Compared to existing RAN slicing schemes for edge computing, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x. Compared to existing RAN slicing frameworks, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x.

Journal extension: multi-cell Top-L association, queueing + Lyapunov control, causal MAB, Sionna PHY tables/traces, RL baselines. Reported **reward** is the Lyapunov objective \(V\cdot u+\mathrm{drift}\) (not the QoS utility alone). Convergence curves/CSVs: `figures/convergence/`.

## Structure
```bash
OMNIS/
│── baselines/                      # RSS, DTS, CTO, GDO, DQN, PPO, MAPPO
│── omnis/                          # Simulator, causal MAB, PHY helpers
│── experiments/
│   ├── convergence_lib.py          # Shared online runner (+ PYCHARM_CONV_*)
│   ├── convergence_all.py / convergence_*.py  # → CSV + series + exports
│   ├── run_sweeps.py / sweep_*.py  # Parameter sweeps → figures/sweeps/
│   ├── plot_results.py             # Convergence PNGs + plot_data.mat/.pkl/.npz
│   ├── plot_sweeps.py              # Sweep PNGs + <name>.mat/.pkl/.npz from CSV
│   └── comm_model.py               # Control-plane bytes / RTT → comm_ms
│── phy_sim/                        # Sionna PHY; see phy_sim/README.md
│── sys_data/
│   ├── config.py                   # Config
│   └── acc_data/                   # Optional seed curves for phy_sim/synth_acc
│── figures/convergence/            # online convergence curves (default)
│── figures/sweeps/                 # sweep outputs (snr/users/arrival/…)
│── observations/                   # Observation figure scripts
```

## How to Try OMNIS

CLI: set `PYTHONPATH=.` and run from the repository root.

**PyCharm:** open any `experiments/convergence_*.py`, `convergence_all.py`, `run_sweeps.py`,
`plot_results.py`, or `plot_sweeps.py` → Run with empty parameters. Edit the
`PYCHARM_*` block at the top of the script (or `PYCHARM_CONV_*` in `convergence_lib.py`).
Scripts `chdir` to the repo root automatically.

### 1. PHY tables / SINR traces
```bash
# See phy_sim/README.md
python3 phy_sim/run_bler.py ...
python3 phy_sim/run_acc.py ...
python3 phy_sim/run_traces.py ...
```

### 2. Convergence for one scheme
```bash
PYTHONPATH=. python3 experiments/convergence_causal.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures/convergence
PYTHONPATH=. python3 experiments/convergence_dqn.py --slots 80 --seeds 0 --out figures/convergence
# same flags for: convergence_ucb / _dts / _gdo / _rss / _ppo / _mappo / _cto
```

### 3. Convergence for selected / all schemes
```bash
# all schemes
PYTHONPATH=. python3 experiments/convergence_all.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures/convergence
# subset (merges into existing perseed.csv; does not wipe other algos)
PYTHONPATH=. python3 experiments/convergence_all.py --algos causal ucb gdo dqn ppo mappo --out figures/convergence
PYTHONPATH=. python3 experiments/convergence_all.py --algos cto --out figures/convergence
```

### 4. Plot convergence results (refreshes exports; keeps CSV/series)
```bash
PYTHONPATH=. python3 experiments/plot_results.py
PYTHONPATH=. python3 experiments/plot_results.py --indir figures/convergence --algos causal ucb cto
```

### 5. Parameter sweeps + re-plot
```bash
PYTHONPATH=. python3 experiments/run_sweeps.py --algos all --only snr users
PYTHONPATH=. python3 experiments/plot_sweeps.py          # PNGs + *.mat/.pkl/.npz from CSV
```

Outputs under `--out` / `--indir` (default `figures/convergence/`):
- `perseed.csv` / `summary.csv` — **Python source** tables (reward, acc, delay, …)
- `series/{name}_seed{k}.npz` — **Python source** time series for curves
- plots from `plot_results.py`:
  - series: `reward.png`, `reward_sliding.png`, `accuracy.png`, `delay.png`,
    `energy.png`, `backlog.png`, `violation.png`
  - mean bars (across seeds, ± std): `reward_bar.png`, `accuracy_bar.png`,
    `delay_bar.png`, `energy_bar.png`, `backlog_bar.png`, `violation_bar.png`
  - one stacked `runtime.png` (**decision** = algo compute, **interaction** =
    control-plane RTT+bytes via `comm_model.py`, **BCD** = resource allocation).
    Old CSVs without `comm_*` derive interaction at plot time. Optional `--users`
    overrides Config.user_num for that derivation.
- `plot_data.mat` + `plot_data.pkl` / `.npz` — MATLAB + Python aggregated exports
- sweeps (`figures/sweeps/<name>/`): `perseed.csv`, metric PNGs,
  `<name>.mat` + `.pkl` / `.npz`

## Algorithms

| Name | Role |
|------|------|
| **causal** | OMNIS-Causal (learns Acc from observations; uninformative prior) |
| **ucb** | OMNIS-UCB (learns reward via GP; random burn-in) |
| **dts** | OMNIS-TS (learns reward via GP; random burn-in) |
| **cto** | Centralized joint GP-UCB (learns reward; random burn-in) |
| **gdo** | Online empirical Acc + SF-ESP Acc-floor greedy (not Acc-table oracle) |
| **rss** | Static random arms |
| **dqn** | Centralized Double-DQN (candidate pool like CTO) |
| **ppo** | Centralized branching PPO |
| **mappo** | Multi-agent PPO (CTDE) |

**Acc table policy:** `mcs_table.accuracy` / `acc_clean` are **environment-only**
(realize observed Acc for rewards/metrics/GP registration). No algorithm may use
channel-conditioned Acc at decision time; all learners start exploratory and learn
Acc or reward from observations. PHY BLER/SE may still drive delay/energy/queue.

## Important Notes
1. Prefer repo root + `PYTHONPATH=.` on the CLI. In PyCharm, empty Parameters is enough
   (scripts call `ensure_repo_root()`).
2. Ensure SINR traces cover at least `config.time_slot_num` slots and enough UEs for `config.user_num`.
3. Hyperparameters live in `sys_data/config.py`. Fair Causal uses `causal_drift_gain=1.0`
   (same `V·u+drift` objective as other schemes); shared knobs (`lyapunov_v`,
   `reward_w_acc`) and Causal method strengths (`causal_beta`,
   `causal_explore_slots`, GP length scales / `causal_gp_signal_var`) may be
   retuned — Acc-table priors are banned (`causal_use_prior=False`).
   GDO uses `gdo_explore_slots` + empirical Acc-floor (`gdo_acc_floor`), not
   offline table Acc. UCB/DTS/CTO use `gp_init_random` for exploratory start.
   CTO remains centralized joint CBO: on-the-fly pool `cto_max_candidates`
   (default 12288; never materialize `(n_models·L)^U`), FastGP predict
   (`cto_use_fast_gp=True`), ARD burn-in then freeze (`cto_gp_burn_in=30`,
   `cto_gp_n_restarts=2`). Decision wall is cut vs the old always-L-BFGS /
   K≈6^6 setup, and is tuned to sit mildly above joint DQN (~1.3–1.6×)
   while ≫ distributed Causal/UCB.
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
