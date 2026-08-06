# OMNIS

## Overview

Edge computing enables resource-constrained devices to execute machine learning applications via task offloading. To this aim, radio access network (RAN) slicing is instrumental to provide the necessary communication and computing resources. However, current RAN slicing approaches rely on static computing models, thereby constraining their ability to leverage the dynamic semantic data representation capabilities enabled by recent neural architectures. In this paper, we propose OMNIS, a semantic RAN slicing framework for edge computing built on a new generation of dynamic split neural models. In contrast to prior work, OMNIS embeds a dynamic form of neural compression paired with adaptive data encoding for task offloading, which provides an ample set of communication payload and computing options for RAN slicing. Differently from prior methodologies for semantic communications, we explicitly study the interplay between neural compression and information quantization in determining the performance of computer vision tasks. In this context, we design a new quantization approach, named ``box'' quantization, which improves resiliency to bit errors as a function of the compression rate compared to current state of the art. Considering the partial observability and differing objectives of the network nodes, we formulate two interdependent optimization problems to achieve optimal inference performance: (i) the mobile devices (MDs) maximize inference accuracy under quality of service (QoS) constraints by controlling the dynamic split deep neural networks (DNNs), and (ii) the edge server (ES) allocates bandwidth and computing resources to maximize the worst inference accuracy among all MDs. To solve these problems, we propose a multi-agent distributed optimization framework, where MDs act as contextual multi-armed bandit (MAB) agents using Bayesian optimization, and the ES performs resource allocation via convex optimization. Compared to existing RAN slicing schemes for edge computing, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x. Compared to existing RAN slicing frameworks, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x.

**Journal extension** adds: multi-cell Top-L association, queueing + Lyapunov DPP, causal MAB (shared residual GP), Sionna PHY tables/traces, and RL baselines (centralized DQN, MAPPO). Main comparison plots/CSVs go to `figures/`.

## Structure
```bash
OMNIS/
│── baselines/                  # Baseline algorithms
│   ├── rss_main.py                 # RSS: static random (model, cell_rank)
│   ├── dts_main.py                 # OMNIS-TS: per-MD GP + Thompson sampling
│   ├── cto_main.py                 # CTO: centralized joint GP-UCB
│   ├── gdo_main.py                 # GDO: SEM-O-RAN SF-ESP greedy (myopic, no DPP)
│   ├── dqn_main.py                 # Centralized branching DQN (global state + team reward)
│   ├── mappo_main.py               # MAPPO: CTDE multi-agent PPO
│   └── rl_slot_env.py              # Shared multi-cell slot loop for DQN / MAPPO
│
│── omnis/                      # Core OMNIS framework
│   ├── omnis_main.py               # Main simulator (algo=causal | ucb)
│   ├── causal_bandit.py            # Shared causal contextual MAB
│   ├── causal_scm.py               # Structural causal model / mechanisms
│   ├── causal_gp.py                # Residual GP with online Cholesky updates
│   ├── fast_gp.py                  # Fast GP predict path (used by CTO / large query sets)
│   ├── cbo.py                      # Contextual BO (UCB / TS) for conference UCB
│   ├── action_space.py             # Action–context space of the MAB agent
│   ├── mcs_table.py                # PHY lookup: BLER / acc / goodput (ILLA)
│   ├── sinr_trace.py               # Multi-cell SINR trace loader (Top-L cells)
│   └── util.py                     # Utility helpers
│
│── scripts/                    # Journal comparison / smoke / training
│   ├── run_multiseed.py            # MAIN: multi-seed long-horizon compare → figures/
│   ├── plot_compare.py             # Single-seed multi-scheme plots → figures/
│   ├── train_dqn.py                # Optional DQN pretrain checkpoint (ablation)
│   ├── smoke_multicell.py          # Quick multi-cell OMNIS smoke
│   ├── smoke_multicell_compare.py  # Short multi-scheme multi-cell smoke
│   ├── smoke_queue_compare.py      # Short queueing + Lyapunov smoke
│   └── validate_queue.py           # Longer queue validation + Lyapunov-V sweep
│
│── experiments/                # Paper-scale evaluation scripts
│   ├── eval_convergence.py         # Per-slot convergence (conference suite)
│   ├── eval_user_num.py            # Sweep user count
│   ├── eval_snr.py                 # Sweep SNR
│   ├── eval_action.py              # Action selection statistics
│   ├── eval_causal_vs_ucb.py       # Causal MAB vs conference GP-UCB
│   ├── eval_causal_ablation.py     # Causal ablations (prior / share / TS)
│   ├── smoke_causal.py             # Tiny causal vs UCB smoke
│   └── *.m                         # MATLAB figure generators (conference)
│
│── phy_sim/                    # Offline Sionna PHY data generation (see phy_sim/README.md)
│   ├── mcs.py / payloads.py / link.py / features.py
│   ├── run_bler.py / run_acc.py / run_traces.py / run_cb_scan.py / synth_acc.py
│   └── output/                     # bler_table, acc_clean, sinr traces, …
│
│── observations/               # Observation / quantization figure scripts (.m)
│── sys_data/
│   ├── config.py                   # System & hyperparameter config
│   ├── acc_data/                   # Legacy accuracy fitting
│   └── mimo_channel_gen/           # Legacy MIMO SNR .npy generators
│── figures/                    # Multiseed plots + CSV (run_multiseed output)
```

## How to Try OMNIS

Set `PYTHONPATH=.` and run from the **repository root**.

### 1. Legacy channel generation (conference pipeline)
```bash
python3 sys_data/mimo_channel_gen/mimo_channel_gen.py
python3 sys_data/mimo_channel_gen/mimo_channel_gen_fix_snr.py
```

### 2. PHY tables / SINR traces (journal; optional if CSVs already in `phy_sim/output/`)
```bash
# See phy_sim/README.md for full workflow
python3 phy_sim/run_bler.py ...
python3 phy_sim/run_acc.py ...
python3 phy_sim/run_traces.py ...
```

### 3. Single-algorithm runs
```bash
PYTHONPATH=. python3 omnis/omnis_main.py          # OMNIS (set config.algo = causal|ucb)
PYTHONPATH=. python3 baselines/rss_main.py
PYTHONPATH=. python3 baselines/dts_main.py
PYTHONPATH=. python3 baselines/cto_main.py
PYTHONPATH=. python3 baselines/gdo_main.py         # SF-ESP greedy (SEM-O-RAN–inspired)
PYTHONPATH=. python3 baselines/dqn_main.py         # centralized branching DQN
PYTHONPATH=. python3 baselines/mappo_main.py       # MAPPO
```

### 4. Journal multi-seed comparison (recommended)
```bash
# All algorithms, 200 slots × 5 seeds → figures/
PYTHONPATH=. python3 scripts/run_multiseed.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures

# Subset only
PYTHONPATH=. python3 scripts/run_multiseed.py --algos causal ucb gdo dqn mappo --out figures

# Single-seed curves
PYTHONPATH=. python3 scripts/plot_compare.py --slots 80 --users 6 --out figures
```

### 5. Smokes / validation
```bash
PYTHONPATH=. python3 scripts/smoke_multicell.py
PYTHONPATH=. python3 scripts/smoke_multicell_compare.py
PYTHONPATH=. python3 scripts/smoke_queue_compare.py
PYTHONPATH=. python3 scripts/validate_queue.py
PYTHONPATH=. python3 experiments/smoke_causal.py
```

### 6. Causal experiments & conference evals
```bash
PYTHONPATH=. python3 experiments/eval_causal_vs_ucb.py
PYTHONPATH=. python3 experiments/eval_causal_ablation.py
PYTHONPATH=. python3 experiments/eval_convergence.py
PYTHONPATH=. python3 experiments/eval_user_num.py
PYTHONPATH=. python3 experiments/eval_snr.py
PYTHONPATH=. python3 experiments/eval_action.py
```

### 7. Optional DQN pretrain (ablation; main compare uses online-from-scratch)
```bash
PYTHONPATH=. python3 scripts/train_dqn.py --slots 400 --seed 0
```

### 8. MATLAB figures
Run the corresponding `.m` scripts under `observations/` and `experiments/`.

## Algorithms (journal compare)

| Name | Role |
|------|------|
| **causal** | OMNIS-Causal: shared residual GP + DPP acquisition |
| **ucb** | OMNIS-UCB: per-MD GP-UCB (conference) |
| **dts** | OMNIS-TS: per-MD GP + Thompson sampling |
| **cto** | Centralized joint GP-UCB |
| **gdo** | Myopic SF-ESP greedy (SEM-O-RAN–inspired; no learning / no DPP) |
| **rss** | Static random arms |
| **dqn** | Centralized branching DQN (global state, team DPP reward) |
| **mappo** | Multi-agent PPO (CTDE) |

## Important Notes
1. Prefer running with the working directory set to the **repo root** and `PYTHONPATH=.`. Some IDEs treat package folders as libraries if misconfigured.
2. Ensure channel / SINR traces cover at least `config.time_slot_num` slots and enough UEs for `config.user_num`.
3. Hyperparameters live in `sys_data/config.py` (queues, Lyapunov \(V\), Top-L cells, DQN/MAPPO/GDO knobs, PHY table paths).
4. When running `eval_user_num.py`, keep `user_num_list` ≤ `config.user_num` after channel generation.
5. This repository does not include training code for the multi-branch dynamic split DNN; it interfaces to evaluation tables. For DNN details, contact Ian Andrew Harshbarger (iharshba@uci.edu).
6. Agent runtime plots report **decision + update** time only (exclude channel estimation and BCD).

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
