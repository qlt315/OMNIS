# OMNIS

## Overview

Edge computing enables resource-constrained devices to execute machine learning applications via task offloading. To this aim, radio access network (RAN) slicing is instrumental to provide the necessary communication and computing resources. However, current RAN slicing approaches rely on static computing models, thereby constraining their ability to leverage the dynamic semantic data representation capabilities enabled by recent neural architectures. In this paper, we propose OMNIS, a semantic RAN slicing framework for edge computing built on a new generation of dynamic split neural models. In contrast to prior work, OMNIS embeds a dynamic form of neural compression paired with adaptive data encoding for task offloading, which provides an ample set of communication payload and computing options for RAN slicing. Differently from prior methodologies for semantic communications, we explicitly study the interplay between neural compression and information quantization in determining the performance of computer vision tasks. In this context, we design a new quantization approach, named ``box'' quantization, which improves resiliency to bit errors as a function of the compression rate compared to current state of the art. Considering the partial observability and differing objectives of the network nodes, we formulate two interdependent optimization problems to achieve optimal inference performance: (i) the mobile devices (MDs) maximize inference accuracy under quality of service (QoS) constraints by controlling the dynamic split deep neural networks (DNNs), and (ii) the edge server (ES) allocates bandwidth and computing resources to maximize the worst inference accuracy among all MDs. To solve these problems, we propose a multi-agent distributed optimization framework, where MDs act as contextual multi-armed bandit (MAB) agents using Bayesian optimization, and the ES performs resource allocation via convex optimization. Compared to existing RAN slicing schemes for edge computing, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x. Compared to existing RAN slicing frameworks, OMNIS improves inference accuracy by up to 22.85\% while reducing the QoS constraint violation probability by up to 10x.

Journal extension: multi-cell Top-L association, queueing + Lyapunov control, causal MAB, Sionna PHY tables/traces, RL baselines. Reported **reward** is the Lyapunov objective \(V\cdot u+\mathrm{drift}\) (not the QoS utility alone). Plots/CSVs: `figures/`.

## Structure
```bash
OMNIS/
│── baselines/
│   ├── rss_main.py                 # RSS
│   ├── dts_main.py                 # OMNIS-TS
│   ├── cto_main.py                 # CTO
│   ├── gdo_main.py                 # GDO
│   ├── dqn_main.py                 # Centralized branching Double-DQN
│   ├── ppo_main.py                 # Centralized branching PPO
│   ├── mappo_main.py               # MAPPO (CTDE)
│   ├── rl_nets.py                  # Shared RL networks
│   └── rl_slot_env.py              # Shared slot loop for DQN / PPO / MAPPO
│
│── omnis/
│   ├── omnis_main.py               # Main simulator
│   ├── causal_bandit.py            # Causal contextual MAB
│   ├── causal_scm.py               # Structural causal model
│   ├── causal_gp.py                # Residual GP
│   ├── fast_gp.py                  # Fast GP predict
│   ├── cbo.py                      # Contextual BO
│   ├── action_space.py             # Action–context space
│   ├── mcs_table.py                # PHY lookup tables
│   ├── sinr_trace.py               # Multi-cell SINR traces
│   └── util.py                     # Utilities
│
│── scripts/
│   ├── train_lib.py                # Shared runner (metrics + plots)
│   ├── train_all.py                # All schemes → figures/
│   ├── train_causal.py             # Per-scheme trainers
│   ├── train_ucb.py / train_dts.py / train_gdo.py / train_rss.py
│   ├── train_dqn.py / train_ppo.py / train_mappo.py / train_cto.py
│
│── experiments/                    # Optional paper sweeps (legacy)
│   ├── eval_convergence.py / eval_user_num.py / eval_snr.py
│   ├── eval_action.py / eval_causal_vs_ucb.py / eval_causal_ablation.py
│   └── *.m
│
│── phy_sim/                        # Sionna PHY data generation; see phy_sim/README.md
│   ├── mcs.py / payloads.py / link.py / features.py
│   ├── run_bler.py / run_acc.py / run_traces.py / run_cb_scan.py / synth_acc.py
│   └── output/
│
│── observations/                   # Observation figure scripts
│── sys_data/
│   ├── config.py                   # Config
│   ├── acc_data/
│   └── mimo_channel_gen/
│── figures/                        # train_*.py output
```

## How to Try OMNIS

Set `PYTHONPATH=.` and run from the repository root.

### 1. Channel generation
```bash
python3 sys_data/mimo_channel_gen/mimo_channel_gen.py
python3 sys_data/mimo_channel_gen/mimo_channel_gen_fix_snr.py
```

### 2. PHY tables / SINR traces
```bash
# See phy_sim/README.md
python3 phy_sim/run_bler.py ...
python3 phy_sim/run_acc.py ...
python3 phy_sim/run_traces.py ...
```

### 3. Train one scheme
```bash
PYTHONPATH=. python3 scripts/train_causal.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures
PYTHONPATH=. python3 scripts/train_dqn.py --slots 80 --seeds 0 --out figures
# same flags for: train_ucb / train_dts / train_gdo / train_rss / train_ppo / train_mappo / train_cto
```

### 4. Train all schemes
```bash
PYTHONPATH=. python3 scripts/train_all.py --slots 200 --users 6 --seeds 0 1 2 3 4 --out figures
PYTHONPATH=. python3 scripts/train_all.py --algos causal ucb gdo dqn ppo mappo --out figures
```

Outputs under `--out` (default `figures/`):
- `summary.csv` / `perseed.csv` — reward, acc, delay, energy, backlog, vio, ms/slot (decision+BCD+update)
- `reward.png`, `accuracy.png`, `delay.png`, `energy.png`, `backlog.png`, `violation.png`
- `runtime.png` (stacked decision/BCD/update), `runtime_log.png`

### 5. Optional paper sweeps
```bash
PYTHONPATH=. python3 experiments/eval_causal_vs_ucb.py
PYTHONPATH=. python3 experiments/eval_causal_ablation.py
PYTHONPATH=. python3 experiments/eval_convergence.py
PYTHONPATH=. python3 experiments/eval_user_num.py
PYTHONPATH=. python3 experiments/eval_snr.py
PYTHONPATH=. python3 experiments/eval_action.py
```

### 7. DQN pretrain
```bash
PYTHONPATH=. python3 scripts/train_dqn.py --slots 400 --seed 0
```

### 8. MATLAB figures
Run the corresponding `.m` scripts under `observations/` and `experiments/`.

## Algorithms

| Name | Role |
|------|------|
| **causal** | OMNIS-Causal |
| **ucb** | OMNIS-UCB |
| **dts** | OMNIS-TS |
| **cto** | Centralized joint GP-UCB |
| **gdo** | SF-ESP greedy |
| **rss** | Static random arms |
| **dqn** | Centralized branching Double-DQN |
| **ppo** | Centralized branching PPO |
| **mappo** | Multi-agent PPO (CTDE) |

## Important Notes
1. Prefer running with the working directory set to the repo root and `PYTHONPATH=.`.
2. Ensure channel / SINR traces cover at least `config.time_slot_num` slots and enough UEs for `config.user_num`.
3. Hyperparameters live in `sys_data/config.py`.
4. When running `eval_user_num.py`, keep `user_num_list` ≤ `config.user_num` after channel generation.
5. This repository does not include training code for the multi-branch dynamic split DNN; it interfaces to evaluation tables. For DNN details, contact Ian Andrew Harshbarger (iharshba@uci.edu).
6. Runtime plots report **decision + BCD + update** wall time per slot.
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
