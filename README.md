# OMNIS+

Journal code for OMNIS+ (branch `journal`, paper `omnis+.tex`).
Multi-cell semantic RAN slicing with a dynamic split DNN, Top-\(L\) cell association,
shared accuracy GP + Lyapunov admission at MDs, and BCD at each ES (bandwidth, GPU, MCS).

## Code layout

```
OMNIS/
├── omnis/           # main simulator
├── baselines/       # comparison schemes
├── experiments/     # run / plot scripts
├── phy_sim/         # PHY tables and channel traces
├── split_dnn/       # split DNN training
├── sys_data/        # config
└── figures/         # plot outputs
```

### `sys_data/`

| File | Description |
|------|-------------|
| `config.py` | All sim hyperparameters: models, payloads, FLOPs, QoS, radio, seeds |

### `omnis/` (online simulator)

| File | Description |
|------|-------------|
| `omnis_main.py` | OMNIS+ agent (causal MAB + BCD); can run one sim standalone |
| `sim_loop.py` | Shared per-slot simulation loop for OMNIS+ and baselines |
| `task_pipeline.py` | MD queues and ES GPU sharing; branch/cell stay fixed until task ends |
| `bcd_loop.py` | Per-slot BCD: bandwidth, GPU frequency, MCS |
| `causal_bandit.py` | Contextual MAB over (branch, cell) with accuracy GP |
| `causal_scm.py` | Maps model / SINR / MCS to accuracy, delay, energy |
| `causal_gp.py` | Shared residual accuracy GP (Matérn-3/2) |
| `cbo.py` | Contextual Bayesian optimization over discrete actions |
| `action_space.py` | Discrete action / context store for CBO |
| `fast_gp.py` | Faster GP predict (used by C-OMNIS+) |
| `util.py` | Helpers for acquisition and CBO utilities |
| `assoc_info.py` | Top-\(L\) cell ranking and compute-queue broadcast |
| `gpu_alloc.py` | Closed-form GPU frequency shares |
| `compute_stats.py` | EWMA delay / energy / work predictors |
| `mcs_table.py` | Load MCS / BLER / Acc CSVs; goodput and Acc lookups |
| `channel_trace.py` | Channel-gain NPZ → IRC SINR |
| `sinr_trace.py` | Load older multi-cell SINR CSV traces |
| `radio_obs.py` | True vs estimated SINR, payload bits, grant timing |
| `exog.py` | Random arrivals, QoS draws, CSI/Acc noise, FLOPs jitter |
| `queue_agent_mixin.py` | Add discrete queues / sticky branch–cell to baselines |
| `proxy_calib.py` | Log predicted vs realized delay/energy (for diagnostics) |

### `baselines/`

| File | Description |
|------|-------------|
| `slot_env.py` | Shared multi-cell slot environment for non-RL baselines |
| `cto_main.py` | C-OMNIS+: same score as OMNIS+, centralized admits |
| `gdo_main.py` | GDO: Acc EWMA+UCB among QoS-feasible arms |
| `dts_main.py` | Scalar GP + Thompson sampling |
| `dqn_main.py` | Centralized Double DQN |
| `ppo_main.py` | Centralized PPO |
| `mappo_main.py` | MAPPO (local actors + central critic) |
| `rl_slot_env.py` | Shared online RL slot loop (DQN / PPO / MAPPO) |
| `rl_nets.py` | Shared RL networks |

(`ucb` / OMNIS is the scalar-GP path inside the causal / CBO stack, run via `experiments/convergence_ucb.py`.)

### `experiments/`

| File | Description |
|------|-------------|
| `convergence_lib.py` | Shared convergence runner → CSV / series / plots |
| `convergence_all.py` | Run convergence for many schemes at once |
| `convergence_causal.py` | Convergence for OMNIS+ |
| `convergence_cto.py` | Convergence for C-OMNIS+ |
| `convergence_ucb.py` | Convergence for OMNIS (scalar GP) |
| `convergence_dts.py` | Convergence for DTS |
| `convergence_gdo.py` | Convergence for GDO |
| `convergence_dqn.py` | Convergence for DQN |
| `convergence_ppo.py` | Convergence for PPO |
| `convergence_mappo.py` | Convergence for MAPPO |
| `run_sweeps.py` | Run parameter sweeps (CLI or `PYCHARM_*` defaults) |
| `sweep_lib.py` | Shared helpers for sweeps |
| `sweep_snr.py` | Metrics vs SNR |
| `sweep_users.py` | Metrics vs number of MDs |
| `sweep_arrival.py` | Metrics vs task arrival rate |
| `sweep_action_pick.py` | Branch pick probability vs SNR / #MDs |
| `sweep_acc_vio.py` | Acc–violation curve (OMNIS+ vs GDO) |
| `plot_results.py` | Plot convergence results |
| `plot_sweeps.py` | Replot sweeps from saved CSV (no re-sim) |
| `plot_proxy_calibration.py` | Plot admit-time delay/energy prediction error |
| `comm_model.py` | Count control-plane bytes → equivalent communication time |
| `result_io.py` | Save/load `.mat` / pickle / npz for plots |
| `repo_util.py` | Put repo root on `sys.path` for IDE runs |
| `launch_overnight_sweeps.py` | Start a long `run_sweeps.py` in the background |
| `train_*.py` / `train_lib.py` | Old names; redirect to `convergence_*` |

### `phy_sim/`

| File | Description |
|------|-------------|
| `mcs.py` | Build 5G NR MCS definition table |
| `payloads.py` | Branch payload sizes (keep in sync with `Config.data_size`) |
| `link.py` | One transport-block encode / transmit / decode (Sionna) |
| `features.py` | Load feature payloads; score Acc with the split-DNN tail |
| `run_bler.py` | Sweep (branch, MCS, SNR) → `bler_table.csv` |
| `run_acc.py` | Sweep (branch, MCS, SNR) → `acc_table.csv` |
| `run_traces.py` | Generate 7-cell UMi channel traces |

Default outputs: `phy_sim/output/` (tag `smoke7`).
Online sim only needs those files; rebuilding needs Sionna (`phy_sim/README.md`).

### `split_dnn/`

Online control never imports this folder. Train models here; payloads go into
`sys_data/config.py`. Details: `split_dnn/README.md`.

| File | Description |
|------|-------------|
| `split_resnet50.py` | Split ResNet-50 head / tail + quantization modes |
| `wrapper_FRCNN.py` | Faster R-CNN teacher / head / tail |
| `wrapper_MaskRCNN.py` | Mask R-CNN teacher / head / tail |
| `boxcoder.py` | Box quantization and bitstream pack / unpack |
| `rpn_updated.py` | Custom RPN for split training |
| `anchor_utils_updated.py` | Anchors for the custom RPN |
| `coco_utils.py` | COCO dataset loaders |
| `transforms.py` | Detection transforms |
| `utils.py` | Training helpers |
| `Split_FRCNN_training.py` | Train split Faster R-CNN |
| `Split_MaskRCNN_training.py` | Train split Mask R-CNN |
| `Split_MaskRCNN_training_with_bitstream.py` | Mask R-CNN + bitstream / BER path |
| `eval_DD.py` | Quick mAP check |
| `bitstream_tests.py` | Bitstream / noise tests |

## Setup

```bash
cd /path/to/OMNIS
export PYTHONPATH=.
# optional: pip install sionna   # only to rebuild phy_sim tables
```

PyCharm: run any `experiments/convergence_*.py` or `run_sweeps.py` with empty
parameters, then edit the `PYCHARM_*` block at the top of the script.
Scripts `chdir` to the repo root.

## How to run

### Convergence

```bash
# one scheme
PYTHONPATH=. python3 experiments/convergence_causal.py --slots 200 --users 6 --seeds 0 1 2 3 4

# several / all
PYTHONPATH=. python3 experiments/convergence_all.py --algos causal ucb gdo dqn ppo mappo cto --slots 200 --users 6 --seeds 0 1 2 3 4
PYTHONPATH=. python3 experiments/convergence_all.py --algos all --slots 200 --users 6 --seeds 0 1 2 3 4

# plots
PYTHONPATH=. python3 experiments/plot_results.py
PYTHONPATH=. python3 experiments/plot_results.py --indir "figures/python figures/convergence" --algos causal ucb cto
```

Output: `figures/python figures/convergence/`
(`perseed.csv`, `summary.csv`, `series/*.npz`, metric PNGs, `runtime.png`).

Reward is the Lyapunov score \(Vu + g^{d}\Delta\) (accuracy + soft QoS + backlog).

### Sweeps

```bash
PYTHONPATH=. python3 experiments/run_sweeps.py --algos all --only snr users
PYTHONPATH=. python3 experiments/plot_sweeps.py
PYTHONPATH=. python3 experiments/sweep_acc_vio.py
```

Also: `sweep_arrival.py`, `sweep_action_pick.py`.
Outputs: `figures/python figures/sweeps/<name>/`.

### Rebuild PHY tables (optional)

```bash
cd phy_sim
python3 mcs.py
python3 run_bler.py ...
python3 run_acc.py ...
python3 run_traces.py ...
```

## Schemes (`--algos`)

| Flag | Paper name | Notes |
|------|------------|--------|
| `causal` | OMNIS+ | shared Acc GP + analytic QoS/drift; Top-\(L\) (branch, cell) |
| `cto` | C-OMNIS+ | same score, centralized sequential controller |
| `ucb` | OMNIS | scalar GP on Lyapunov reward + UCB |
| `dts` | — | scalar GP + Thompson sampling |
| `gdo` | GDO | Acc EWMA+UCB among QoS-feasible arms; no drift term |
| `dqn` / `ppo` / `mappo` | DQN / PPO / MAPPO | DRL baselines; same ES BCD |

Always run from the repo root with `PYTHONPATH=.`.
SINR traces must cover `time_slot_num` slots and enough UEs for `user_num`.
Runtime bar = decision + MD–ES interaction (`comm_model.py`) + BCD.

GP-UCB follows https://github.com/jaayala/contextual_bayesian_optimization
