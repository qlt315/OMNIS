# Parameter sweeps

Online re-simulation for each (algo, seed, setting). No reuse of convergence checkpoints.

| Sweep | Script | Axis |
|-------|--------|------|
| SNR | `sweep_snr.py` | best-cell SINR via `sinr_offset_db` |
| Users | `sweep_users.py` | #MDs |
| Arrival | `sweep_arrival.py` | Poisson λ |
| Action pick | `sweep_action_pick.py` | model pick vs SNR / #MDs |
| Acc–vio | `sweep_acc_vio.py` | Acc–vio Pareto: OMNIS+ vs GDO (`feas_margin`) |

```bash
PYTHONPATH=. python3 experiments/run_sweeps.py
PYTHONPATH=. python3 experiments/run_sweeps.py --algos causal gdo --only snr users
PYTHONPATH=. python3 experiments/sweep_acc_vio.py
PYTHONPATH=. python3 experiments/plot_sweeps.py
```

Defaults: slots=500, seeds 0–2, all algos. Edit `PYCHARM_*` in `run_sweeps.py` for PyCharm.
