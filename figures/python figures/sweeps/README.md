# Parameter sweeps

Online re-simulation for each (algo, seed, setting). No reuse of convergence checkpoints.

| Sweep | Script | Axis |
|-------|--------|------|
| SNR | `sweep_snr.py` | best-cell SINR via `sinr_offset_db` |
| Users | `sweep_users.py` | #MDs [5, 10, 15, 20, 25] |
| Arrival | `sweep_arrival.py` | Poisson λ |
| Action pick | `sweep_action_pick.py` | model pick vs SNR / #MDs |
| Acc–vio | `sweep_acc_vio.py` | per-algo QoS knob (Pareto) |

```bash
PYTHONPATH=. python3 experiments/run_sweeps.py
PYTHONPATH=. python3 experiments/run_sweeps.py --algos causal gdo --only snr users
PYTHONPATH=. python3 experiments/sweep_acc_vio.py
PYTHONPATH=. python3 experiments/plot_sweeps.py
```

Defaults: slots=500, users=10 (SNR/arrival), seeds 0–2,
algos=all (causal, ucb, dts, gdo, dqn, ppo, mappo, cto). Edit `PYCHARM_*` in `run_sweeps.py` for PyCharm.
