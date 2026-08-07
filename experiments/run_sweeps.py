#!/usr/bin/env python3
"""Overnight orchestrator for OMNIS parameter sweeps.

Re-runs online simulation for every setting (no fake checkpoints).
Focus algos: causal, ucb, gdo, dqn, ppo (override with --algos / --all).

Examples:
  PYTHONPATH=. python3 experiments/run_sweeps.py --smoke
  PYTHONPATH=. python3 experiments/run_sweeps.py --algos causal ucb gdo
  PYTHONPATH=. python3 experiments/run_sweeps.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_lib import (
    ALGO_NAMES,
    DEFAULT_SLOTS,
    DEFAULT_SWEEP_ALGOS,
    DEFAULT_USER_LIST,
    DEFAULT_USERS,
    TRACE_MEAN_BEST_CELL_SINR_DB,
    SWEEP_UE_POOL_SIZE,
    sweep_action_pick,
    sweep_arrival,
    sweep_snr,
    sweep_users,
)


def log(msg, fp=None):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None and fp is not sys.stdout:
        fp.write(line + "\n")
        fp.flush()


def write_readme(out_root):
    path = os.path.join(out_root, "README.md")
    text = f"""# OMNIS parameter sweeps

Honest **online re-simulation** for every (algo, seed, setting). Checkpoints from
`train_all` are **not** reused: MAB schemes never save GPs; DQN only writes
`figures/dqn_pretrained.pt` via explicit `pretrain()` (not called by `train_all`);
PPO/MAPPO do not save.

## Paper alignment (Sec. VI)

| Sweep | Script | Paper figure | Axis |
|-------|--------|--------------|------|
| SNR | `sweep_snr.py` | Fig. 6 | target mean **best-cell** SINR 0–10 dB via `sinr_offset_db` |
| Users | `sweep_users.py` | Fig. 7 | #MDs {list(DEFAULT_USER_LIST)}; nested prefixes of {SWEEP_UE_POOL_SIZE}-UE pool |
| Action pick | `sweep_action_pick.py` | Fig. 8 | pick hist @ SNR∈{{2,4,6}}, MDs∈{{10,15,20}} + β sweep |
| Arrival | `sweep_arrival.py` | (journal) | Poisson λ tasks/slot |

## SNR axis definition

`sinr_offset_db = snr_target_db − TRACE_MEAN_BEST_CELL_SINR_DB` (≈{TRACE_MEAN_BEST_CELL_SINR_DB:.2f} dB on
the fixed {SWEEP_UE_POOL_SIZE}-UE pool of `smoke7_sites`). The labeled SNR tracks mean max-cell /
serving SINR used by association and MCS — **not** the all-cell mean (≈−10.75 dB).

## GDO

GDO is **SF-ESP Acc-floor greedy** (SEM-O-RAN / Puligheddu TMC 2024 spirit):
lightest model with offline `a(z) ≥ gdo_acc_floor` (default **0.25**), then
best-cell + offer/price knapsack EG. Not a V·u+drift / DPP oracle.

## Metrics

- **reward** — mean Lyapunov `V·u + drift` (same as `train_lib`)
- **delay** — mean latency [s] (log-y plots when helpful)
- **energy** — mean energy [J]
- **acc** — mean inference accuracy
- **vio** — QoS violation probability (delay **or** energy); paper “Avg. Violation Prob.”
- **backlog** — mean queue backlog [bits] (log-y plots when helpful)

## Outputs

Each sweep writes under `figures/sweeps/<name>/`:

- `*_vs_*.png` — metric vs sweep axis (mean over seeds; no error bars)
- `action_pick_*.png` — model selection histograms
- `<name>.mat` — MATLAB arrays: `axis`, `{{algo}}_{{metric}}_mean/std`, `raw_*`
- `perseed.csv` — one row per (algo, seed, axis value)

## Run

```bash
MPLBACKEND=Agg PYTHONPATH=. python3 experiments/run_sweeps.py
PYTHONPATH=. python3 experiments/sweep_snr.py --algos causal ucb gdo --seeds 0 1 2
```

Defaults: slots={DEFAULT_SLOTS}, users={DEFAULT_USERS} for SNR/arrival, seeds 0–2,
algos causal/ucb/gdo/dqn/ppo, users axis {list(DEFAULT_USER_LIST)}.
`causal_drift_gain=1.0` (fair).

Log: `figures/sweeps/overnight.log`. Status: `figures/sweeps/STATUS.md`.
"""
    with open(path, "w") as f:
        f.write(text)
    return path


def sweep_complete(out_root, name):
    if name == "action_pick":
        return os.path.isfile(os.path.join(out_root, name, "action_pick_snr.mat"))
    return os.path.isfile(os.path.join(out_root, name, f"{name}.mat"))


def write_status(out_root, finished, failed, notes):
    path = os.path.join(out_root, "STATUS.md")
    lines = [
        "# Sweep STATUS",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Model saving answer",
        "",
        "- After `train_all`, **MAB models are not saved** (Causal/UCB/DTS/CTO GPs "
        "live only in-process).",
        "- **DQN** can `torch.save` via `save_pretrained()` only after `pretrain()`; "
        "`train_all` → `simulation()` does **not** save; no `figures/*.pt` from Train A.",
        "- **PPO / MAPPO** do not persist weights.",
        "- Sweeps therefore **re-run online simulation** with fixed seeds (honest).",
        "",
        "## Finished",
        "",
    ]
    for item in finished:
        lines.append(f"- {item}")
    if failed:
        lines.extend(["", "## Failed", ""])
        for item in failed:
            lines.append(f"- {item}")
    lines.extend(["", "## Notes", ""])
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algos", nargs="+", default=list(DEFAULT_SWEEP_ALGOS))
    p.add_argument("--all", action="store_true",
                   help="Use all ALGOS (includes heavy CTO)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    p.add_argument("--users", type=int, default=DEFAULT_USERS,
                   help="Default MD count for SNR/arrival sweeps")
    p.add_argument("--out-root", default="figures/sweeps")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny grid: 30 slots, 1 seed, few axis points")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["snr", "users", "arrival", "action_pick"],
                   help="Skip named sweeps")
    p.add_argument("--resume", action="store_true",
                   help="Skip sweeps that already have complete .mat outputs")
    args = p.parse_args()

    algos = list(ALGO_NAMES) if args.all else list(args.algos)
    seeds = tuple(args.seeds)
    slots = args.slots
    users = args.users
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    write_readme(out_root)

    if args.smoke:
        seeds = (0,)
        slots = 30
        users = min(users, 10)
        snr_targets = [0, 6, 10]
        user_list = [5, 15, 25]
        rates = [0.4, 0.7, 1.0]
        pick_snr = [2, 6]
        pick_users = [10, 20]
        betas = [0.55, 1.0]
    else:
        snr_targets = [0, 2, 4, 6, 8, 10]
        user_list = list(DEFAULT_USER_LIST)
        rates = [0.30, 0.50, 0.70, 0.90, 1.10]
        pick_snr = [2, 4, 6]
        pick_users = [10, 15, 20]
        betas = [0.55, 1.0, 2.0]

    finished, failed, notes = [], [], []
    notes.append(f"algos={algos} seeds={seeds} slots={slots} users={users}")
    notes.append(
        f"users axis={user_list}; UE pool={SWEEP_UE_POOL_SIZE} (stable prefix); "
        f"TRACE_MEAN_BEST_CELL_SINR_DB≈{TRACE_MEAN_BEST_CELL_SINR_DB:.2f} "
        f"(calibrated on {SWEEP_UE_POOL_SIZE}-UE pool)"
    )
    notes.append(
        "sinr_offset_db = snr_target_db - TRACE_MEAN_BEST_CELL_SINR_DB; "
        "axis = mean best-cell / serving SINR (not all-cell mean)"
    )
    notes.append(
        "plots: mean lines only (no error bars); log-y for delay/backlog; "
        "causal_drift_gain=1.0; GDO=SF-ESP Acc-floor greedy (gdo_acc_floor=0.25)"
    )

    skip = set(args.skip)
    if args.resume:
        for name in ("snr", "users", "arrival", "action_pick"):
            if sweep_complete(out_root, name):
                skip.add(name)
                notes.append(f"resume: skip complete `{name}`")

    t0 = time.time()
    log(f"START sweeps smoke={args.smoke} algos={algos} skip={sorted(skip)}")

    jobs = []
    if "snr" not in skip:
        jobs.append(("snr", lambda: sweep_snr(
            algos=algos, seeds=seeds, slots=slots, users=users,
            snr_targets=snr_targets, out_root=out_root)))
    if "users" not in skip:
        jobs.append(("users", lambda: sweep_users(
            algos=algos, seeds=seeds, slots=slots,
            user_list=user_list, snr_db=5.0, out_root=out_root)))
    if "arrival" not in skip:
        jobs.append(("arrival", lambda: sweep_arrival(
            algos=algos, seeds=seeds, slots=slots, users=users,
            rates=rates, snr_db=5.0, out_root=out_root)))
    if "action_pick" not in skip:
        jobs.append(("action_pick", lambda: sweep_action_pick(
            algos=algos, seeds=seeds, slots=slots,
            snr_targets=pick_snr, user_list=pick_users,
            beta_values=betas, out_root=out_root)))

    for name, fn in jobs:
        log(f"=== begin {name} ===")
        try:
            fn()
            finished.append(f"`{name}` → `{out_root}/{name}/`")
            log(f"=== finished {name} ===")
        except Exception as e:
            failed.append(f"`{name}`: {e}")
            log(f"=== FAILED {name}: {e} ===")
            traceback.print_exc()
        write_status(out_root, finished, failed, notes)

    elapsed = time.time() - t0
    notes.append(f"total wall {elapsed/3600:.2f} h")
    for sweep in ("snr", "users", "arrival", "action_pick"):
        d = os.path.join(out_root, sweep)
        if os.path.isdir(d):
            arts = sorted(os.listdir(d))
            notes.append(f"{sweep} artifacts: {', '.join(arts)}")
    write_status(out_root, finished, failed, notes)
    log(f"DONE wall={elapsed/60:.1f} min finished={finished} failed={failed}")


if __name__ == "__main__":
    main()
