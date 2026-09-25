# OMNIS

## Overview

Edge computing enables resource-constrained devices to execute machine learning applications via task offloading. To this aim, radio access network (RAN) slicing is instrumental to provide the necessary communication and computing resources. However, current RAN slicing approaches rely on static computing models, thereby constraining their ability to leverage the dynamic semantic data representation capabilities enabled by recent neural architectures. In this paper, we propose OMNIS, a semantic RAN slicing framework for edge computing built on a new generation of dynamic split neural models. In contrast to prior work, OMNIS embeds a dynamic form of neural compression paired with adaptive data encoding for task offloading, which provides an ample set of communication payload and computing options for RAN slicing. Differently from prior methodologies for semantic communications, we explicitly study the interplay between neural compression and information quantization in determining the performance of computer vision tasks. In this context, we design a new quantization approach, named ``box'' quantization, which improves resiliency to bit errors as a function of the compression rate compared to current state of the art. Considering the partial observability and differing objectives of the network nodes, we formulate two interdependent optimization problems to achieve optimal inference performance: (i) the mobile devices (MDs) maximize inference accuracy under quality of service (QoS) constraints by controlling the dynamic split deep neural networks (DNNs), and (ii) the edge server (ES) allocates bandwidth and computing resources to maximize the worst inference accuracy among all MDs. To solve these problems, we propose a multi-agent distributed optimization framework, where MDs act as contextual multi-armed bandit (MAB) agents using Bayesian optimization, and the ES performs resource allocation via convex optimization. Compared to existing RAN slicing schemes for edge computing, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x. Compared to existing RAN slicing frameworks, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x.

Journal extension: multi-cell Top-L association, queueing + Lyapunov control, causal MAB, Sionna PHY tables/traces, RL baselines. Reported **reward** is the Lyapunov objective \(V\cdot u+\mathrm{drift}\) (not the QoS utility alone). Convergence curves/CSVs: `figures/python figures/convergence/`.

## Structure
```bash
OMNIS/
│── baselines/                      # DTS, CTO, GDO, DQN, PPO, MAPPO
│── omnis/                          # Simulator, causal MAB, PHY helpers
│── experiments/
│   ├── convergence_lib.py          # Shared online runner (+ PYCHARM_CONV_*)
│   ├── convergence_all.py / convergence_*.py  # → CSV + series + exports
│   ├── run_sweeps.py / sweep_*.py  # Parameter sweeps → figures/python figures/sweeps/
│   ├── plot_results.py             # Convergence PNGs + plot_data.mat/.pkl/.npz
│   ├── plot_sweeps.py              # Sweep PNGs + <name>.mat/.pkl/.npz from CSV
│   └── comm_model.py               # Control-plane bytes / RTT → comm_ms
│── phy_sim/                        # Sionna PHY; see phy_sim/README.md
│── sys_data/
│   ├── config.py                   # Config (points at phy_sim/output)
│   └── acc_data/                   # Optional seed curves for phy_sim/synth_acc
│── figures/
│   ├── python figures/
│   │   ├── convergence/            # online convergence curves (default)
│   │   └── sweeps/                 # snr / users / arrival / action_pick / acc_vio
│   ├── matlab figures/             # exported .eps / .fig
│   └── matlab scripts/             # MATLAB figure generators (from *.mat)
│── observations/                   # Observation figure scripts
```

## How to Try OMNIS

CLI: set `PYTHONPATH=.` and run from the repository root.

**PyCharm:** open any `experiments/convergence_*.py`, `convergence_all.py`, `run_sweeps.py`,
`plot_results.py`, or `plot_sweeps.py` → Run with empty parameters. Edit the
`PYCHARM_*` block at the top of the script (or `PYCHARM_CONV_*` in `convergence_lib.py`).
Scripts `chdir` to the repo root automatically.

### 1. PHY tables / SINR traces (Sionna)
Link-level MCS / BLER / Acc tables and multi-cell SINR traces are generated under
`phy_sim/` with [Sionna](https://nvlabs.github.io/sionna/) (5G NR LDPC, QAM, 3GPP 38.901).
Outputs land in `phy_sim/output/` (`mcs_def.csv`, `bler_table.csv`, `acc_table.csv`,
`channel_*.npz`, …). The system simulator loads them via `omnis/mcs_table.py` and
`omnis/sinr_trace.py` (`Config.sinr_trace_dir` / `sinr_trace_tag`, default tag `smoke7`).

```bash
# Full pipeline and flags: phy_sim/README.md
cd phy_sim
python3 mcs.py                  # → output/mcs_def.csv
python3 run_bler.py ...         # → output/bler_table.csv
python3 run_acc.py ...          # → output/acc_table.csv, acc_clean.csv
python3 run_traces.py ...       # → output/channel_*.npz
```

### 2. Convergence for one scheme
```bash
PYTHONPATH=. python3 experiments/convergence_causal.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out "figures/python figures/convergence"
PYTHONPATH=. python3 experiments/convergence_dqn.py --slots 80 --seeds 0 --out "figures/python figures/convergence"
# same flags for: convergence_ucb / _dts / _gdo / _ppo / _mappo / _cto
```

### 3. Convergence for selected / all schemes
```bash
# all schemes
PYTHONPATH=. python3 experiments/convergence_all.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out "figures/python figures/convergence"
# subset (merges into existing perseed.csv; does not wipe other algos)
PYTHONPATH=. python3 experiments/convergence_all.py --algos causal ucb gdo dqn ppo mappo --out "figures/python figures/convergence"
PYTHONPATH=. python3 experiments/convergence_all.py --algos cto --out "figures/python figures/convergence"
```

### 4. Plot convergence results (refreshes exports; keeps CSV/series)
```bash
PYTHONPATH=. python3 experiments/plot_results.py
PYTHONPATH=. python3 experiments/plot_results.py --indir "figures/python figures/convergence" --algos causal ucb cto
```

### 5. Parameter sweeps + re-plot
```bash
PYTHONPATH=. python3 experiments/run_sweeps.py --algos all --only snr users
PYTHONPATH=. python3 experiments/plot_sweeps.py          # PNGs + *.mat/.pkl/.npz from CSV
# Acc–vio Pareto (shared env; per-algo QoS knob)
PYTHONPATH=. python3 experiments/sweep_acc_vio.py
```

Outputs under `--out` / `--indir` (default `figures/python figures/convergence/`):
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
- sweeps (`figures/python figures/sweeps/<name>/`): `perseed.csv`, metric PNGs,
  `<name>.mat` + `.pkl` / `.npz`

## Algorithms

| Name | Role |
|------|------|
| **causal** | OMNIS-Causal (learns Acc from observations; uninformative prior) |
| **ucb** | OMNIS-UCB (learns reward via GP; random burn-in) |
| **dts** | OMNIS-TS (learns reward via GP; random burn-in) |
| **cto** | Centralized causal: same score, joint cell-bandwidth share, sequential controller |
| **gdo** | SEM-O-RAN: max Acc-UCB among QoS-feasible (branch, cell) arms; ES BCD for BW/GPU |
| **dqn** | Centralized Double-DQN (joint candidate pool) |
| **ppo** | Centralized branching PPO |
| **mappo** | Multi-agent PPO (CTDE) |

**Acc table policy:** channel-conditioned `mcs_table.accuracy` is environment-only
(realize observed Acc for rewards, metrics, and GP labels). No learner may read it
at decision time. GDO follows SEM-O-RAN without an Acc floor: among
(branch, cell) that meet this task's delay and energy it takes the largest Acc-UCB
and the ES runs BCD for resources. The offline Acc curve is not read at admission.
PHY BLER/SE (from Sionna tables) may still drive delay, energy, and queues.

## Important Notes
1. Prefer repo root + `PYTHONPATH=.` on the CLI. In PyCharm, empty Parameters is enough
   (scripts call `ensure_repo_root()`).
2. Ensure SINR traces under `phy_sim/output/` cover at least `config.time_slot_num` slots
   and enough UEs for `config.user_num` (see `Config.sinr_trace_tag`).
3. Hyperparameters live in `sys_data/config.py`. Fair Causal uses `causal_drift_gain=1.0`
   (same `V·u+drift` objective as other schemes); shared knobs (`lyapunov_v`,
   `reward_w_acc`, `reward_qos_coef`) and Causal method strengths (`causal_beta`,
   `causal_explore_slots`, `causal_feas_margin`, GP length scales /
   `causal_gp_signal_var`) may be retuned — channel-conditioned Acc priors are banned
   (`causal_use_prior=False`).    GDO uses Acc-UCB with optional `gdo_feas_margin`
   (default 1.0). UCB/DTS use `gp_init_random` for exploratory start.
   CTO is centralized causal: the same residual accuracy GP, Lyapunov score,
   and last-slot bandwidth as OMNIS+. MDs that select the same cell are
   rescored on that cell with the pool split, never above the bandwidth
   already observed, and are not moved to an idle cell.
   Decision time is the sum of per-MD work. The control uplink carries
   observations rather than a local action.
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
