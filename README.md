# OMNIS+

Journal code for **OMNIS+**: multi-cell semantic RAN slicing with dynamic split DNNs
(box / standard quantization), Top-$L$ cell association, shared accuracy GP,
Lyapunov admission at MDs, and BCD resource allocation (bandwidth, GPU, MCS) at each ES.

Conference OMNIS is the single-cell predecessor; this tree is the multi-cell extension.

## Layout

```
OMNIS/
├── omnis/           # simulator, causal MAB, BCD, PHY loaders
├── baselines/       # ucb, dts, cto, gdo, dqn, ppo, mappo
├── experiments/     # convergence_*, train_*, sweep_*, plot_*
├── phy_sim/         # Sionna MCS / BLER / Acc tables + SINR traces
├── split_dnn/       # offline Mask R-CNN split training (not imported online)
├── sys_data/        # Config + optional Acc seed curves
└── figures/         # python / matlab plot outputs
```

## Setup

```bash
cd /path/to/OMNIS
export PYTHONPATH=.
# optional: pip install sionna   # only for regenerating phy_sim tables
```

Hyperparameters: `sys_data/config.py` (`lyapunov_v`, `causal_*`, `gdo_feas_margin`,
`sinr_trace_tag`, …). Traces default to `phy_sim/output/` with tag `smoke7`.

PyCharm: run any `experiments/convergence_*.py` or `run_sweeps.py` with empty
parameters; edit the `PYCHARM_*` / `PYCHARM_CONV_*` block at the top of the script.
Scripts `chdir` to the repo root.

## PHY tables (optional rebuild)

See `phy_sim/README.md`. Online sim only needs CSVs / `channel_*.npz` under
`phy_sim/output/` (`mcs_def.csv`, `bler_table.csv`, `acc_table.csv`, …).

```bash
cd phy_sim
python3 mcs.py
python3 run_bler.py ...
python3 run_acc.py ...
python3 run_traces.py ...
```

`acc_table.csv` is used to **realize** completion accuracy in the environment.
Admission / MCS scoring does not look it up; accuracy forecasts come from the
shared GP (OMNIS+ / CTO) or from completion EWMA+UCB (GDO).

## Convergence

```bash
# one scheme
PYTHONPATH=. python3 experiments/convergence_causal.py --slots 200 --users 6 --seeds 0 1 2 3 4

# several / all
PYTHONPATH=. python3 experiments/convergence_all.py --algos causal ucb gdo dqn ppo mappo cto --slots 200 --users 6 --seeds 0 1 2 3 4
PYTHONPATH=. python3 experiments/convergence_all.py --algos all --slots 200 --users 6 --seeds 0 1 2 3 4

# plots (PNG + plot_data.mat / .pkl / .npz)
PYTHONPATH=. python3 experiments/plot_results.py
PYTHONPATH=. python3 experiments/plot_results.py --indir "figures/python figures/convergence" --algos causal ucb cto
```

Default output dir: `figures/python figures/convergence/`
(`perseed.csv`, `summary.csv`, `series/*.npz`, metric PNGs, stacked `runtime.png`).

Reported reward is the Lyapunov score \(V u + g^{d}\Delta\) (accuracy + soft QoS + backlog), not accuracy alone.

## Sweeps

```bash
PYTHONPATH=. python3 experiments/run_sweeps.py --algos all --only snr users
PYTHONPATH=. python3 experiments/plot_sweeps.py
PYTHONPATH=. python3 experiments/sweep_acc_vio.py
```

Outputs under `figures/python figures/sweeps/<name>/`.

## Schemes (`--algos`)

| Flag | Paper name | Notes |
|------|------------|--------|
| `causal` | OMNIS+ | shared Acc GP + analytic QoS/drift; Top-$L$ (branch, cell) |
| `cto` | C-OMNIS+ | same score, centralized sequential controller |
| `ucb` | OMNIS | scalar GP on Lyapunov reward + UCB (ablation) |
| `dts` | — | scalar GP + Thompson sampling (ablation) |
| `gdo` | GDO | Acc EWMA+UCB among QoS-feasible arms; no drift term |
| `dqn` / `ppo` / `mappo` | DQN / PPO / MAPPO | DRL baselines; same ES BCD |

## Notes

1. Run from repo root with `PYTHONPATH=.`.
2. SINR traces must cover `time_slot_num` slots and enough UEs for `user_num`.
3. Online control does not import `split_dnn/`; payloads come from `Config.data_size`.
4. Runtime bar = decision + interaction (`comm_model.py`) + BCD.

## Acknowledgments

Multi-agent contextual GP-UCB follows
https://github.com/jaayala/contextual_bayesian_optimization
