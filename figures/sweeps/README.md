# OMNIS parameter sweeps

Honest **online re-simulation** for every (algo, seed, setting). Checkpoints from
`convergence_all` are **not** reused.

## Outputs

Each sweep writes under `figures/sweeps/<name>/`:

- `perseed.csv` — **Python source** (kept across replot)
- `*_vs_*.png` — metric vs sweep axis
- `<name>.mat` + `<name>.pkl` / `.npz` — aggregated exports (MATLAB + Python)
- action_pick also has `perseed_{snr,users,explore}.csv` and pick mats

Re-plot without re-sim: `PYTHONPATH=. python3 experiments/plot_sweeps.py`

## Run

```bash
MPLBACKEND=Agg PYTHONPATH=. python3 experiments/run_sweeps.py
PYTHONPATH=. python3 experiments/sweep_snr.py --algos causal ucb gdo --seeds 0 1 2
```
