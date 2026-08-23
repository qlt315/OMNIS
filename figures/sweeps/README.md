# OMNIS parameter sweeps

Honest **online re-simulation** for every (algo, seed, setting). Checkpoints from
`convergence_all` are **not** reused.

## Paper alignment (Sec. VI)

| Sweep | Script | Paper figure | Axis |
|-------|--------|--------------|------|
| SNR | `sweep_snr.py` | Fig. 6 | target mean **best-cell** SINR 0–10 dB via `sinr_offset_db` |
| Users | `sweep_users.py` | Fig. 7 | #MDs [5, 10, 15, 20, 25]; nested prefixes of 25-UE pool |
| Action pick | `sweep_action_pick.py` | Fig. 8 | pick hist @ SNR∈{2,4,6}, MDs∈{10,15,20} + β sweep |
| Arrival | `sweep_arrival.py` | (journal) | Poisson λ tasks/slot |

## Run (PyCharm or CLI)

```bash
# Full suite, all algorithms (default)
PYTHONPATH=. python3 experiments/run_sweeps.py

# Subset of algorithms
PYTHONPATH=. python3 experiments/run_sweeps.py --algos causal ucb gdo dts

# Only some sweeps
PYTHONPATH=. python3 experiments/run_sweeps.py --only snr users
```

In PyCharm: edit ``PYCHARM_ALGOS`` / ``PYCHARM_ONLY`` at the top of
``experiments/run_sweeps.py``, then Run with empty parameters.

Defaults: slots=500, users=10 for SNR/arrival, seeds 0–2,
algos=all (causal, ucb, dts, gdo, rss, dqn, ppo, mappo, cto), users axis [5, 10, 15, 20, 25].

Log: `figures/sweeps/overnight.log`. Status: `figures/sweeps/STATUS.md`.
