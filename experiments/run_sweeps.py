#!/usr/bin/env python3
"""Overnight orchestrator for OMNIS parameter sweeps.

Re-runs online simulation for every setting (no fake checkpoints).

PyCharm: open this file → Run (no parameters needed). Edit the
``PYCHARM_*`` block below to choose algorithms / which sweeps to run.

CLI examples:
  PYTHONPATH=. python3 experiments/run_sweeps.py
  PYTHONPATH=. python3 experiments/run_sweeps.py --algos causal ucb gdo
  PYTHONPATH=. python3 experiments/run_sweeps.py --algos all --only snr users
  PYTHONPATH=. python3 experiments/run_sweeps.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

# Repo root as CWD so phy_sim / figures / relative paths work from PyCharm.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

from sweep_lib import (  # noqa: E402
    ALGO_NAMES,
    DEFAULT_ARRIVAL_RATES,
    DEFAULT_SLOTS,
    DEFAULT_SWEEP_ALGOS,
    DEFAULT_USER_LIST,
    DEFAULT_USERS,
    TRACE_MEAN_BEST_CELL_SINR_DB,
    SWEEP_UE_POOL_SIZE,
    resolve_algos,
    sweep_action_pick,
    sweep_arrival,
    sweep_snr,
    sweep_users,
)

# =============================================================================
# PyCharm / zero-arg defaults — edit here, then Run without CLI parameters.
# =============================================================================
# Algorithms: None or "all" → every scheme; or a subset list, e.g.
#   ["causal", "ucb", "gdo", "dqn", "ppo"]
PYCHARM_ALGOS = None  # None | "all" | ["causal", "ucb", ...]

# Which sweeps to run. If ONLY is non-empty it wins; else SKIP is applied.
# Names: "snr", "users", "arrival", "action_pick"
PYCHARM_ONLY = []          # e.g. ["snr", "users"]
PYCHARM_SKIP = []          # e.g. ["action_pick"]

PYCHARM_SEEDS = [0, 1, 2]
PYCHARM_SLOTS = DEFAULT_SLOTS
PYCHARM_USERS = DEFAULT_USERS  # MD count for SNR / arrival axes
PYCHARM_SMOKE = False          # True → tiny grid for a dry run
PYCHARM_OUT_ROOT = "figures/sweeps"
PYCHARM_RESUME = False         # skip sweeps that already have .mat outputs
# =============================================================================

SWEEP_NAMES = ("snr", "users", "arrival", "action_pick")


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
`convergence_all` are **not** reused.

## Paper alignment (Sec. VI)

| Sweep | Script | Paper figure | Axis |
|-------|--------|--------------|------|
| SNR | `sweep_snr.py` | Fig. 6 | target mean **best-cell** SINR 0–10 dB via `sinr_offset_db` |
| Users | `sweep_users.py` | Fig. 7 | #MDs {list(DEFAULT_USER_LIST)}; nested prefixes of {SWEEP_UE_POOL_SIZE}-UE pool |
| Action pick | `sweep_action_pick.py` | Fig. 8 | pick hist @ SNR∈{{2,4,6}}, MDs∈{{10,15,20}} + β sweep |
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

Defaults: slots={DEFAULT_SLOTS}, users={DEFAULT_USERS} for SNR/arrival, seeds 0–2,
algos=all ({', '.join(ALGO_NAMES)}), users axis {list(DEFAULT_USER_LIST)}.

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


def _normalize_algo_spec(spec):
    """None / 'all' / list → validated algo name list."""
    if spec is None or spec == "all":
        return list(DEFAULT_SWEEP_ALGOS)
    if isinstance(spec, str):
        spec = [spec]
    return resolve_algos(list(spec))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--algos", nargs="+", default=None, metavar="NAME",
        help=("schemes to run. Pass 'all' or omit for the full set "
              f"({', '.join(ALGO_NAMES)}). Example: --algos causal ucb gdo"),
    )
    p.add_argument("--all", action="store_true",
                   help="Force all algorithms (same as --algos all)")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--slots", type=int, default=None)
    p.add_argument("--users", type=int, default=None,
                   help="Default MD count for SNR/arrival sweeps")
    p.add_argument("--out-root", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny grid: 30 slots, 1 seed, few axis points")
    p.add_argument("--only", nargs="+", default=None,
                   choices=list(SWEEP_NAMES),
                   help="Run only these sweeps (overrides --skip / PYCHARM_SKIP)")
    p.add_argument("--skip", nargs="+", default=None,
                   choices=list(SWEEP_NAMES),
                   help="Skip named sweeps")
    p.add_argument("--resume", action="store_true",
                   help="Skip sweeps that already have complete .mat outputs")
    args = p.parse_args(argv)

    # Merge CLI over PyCharm defaults (CLI wins when provided).
    if args.all or (args.algos is not None and args.algos == ["all"]):
        algos = list(ALGO_NAMES)
    elif args.algos is not None:
        algos = resolve_algos(args.algos)
    else:
        algos = _normalize_algo_spec(PYCHARM_ALGOS)

    seeds = tuple(args.seeds if args.seeds is not None else PYCHARM_SEEDS)
    slots = args.slots if args.slots is not None else PYCHARM_SLOTS
    users = args.users if args.users is not None else PYCHARM_USERS
    out_root = args.out_root if args.out_root is not None else PYCHARM_OUT_ROOT
    smoke = bool(args.smoke or PYCHARM_SMOKE)
    resume = bool(args.resume or PYCHARM_RESUME)

    if args.only is not None:
        only = set(args.only)
    elif PYCHARM_ONLY:
        only = set(PYCHARM_ONLY)
    else:
        only = set()

    if args.skip is not None:
        skip = set(args.skip)
    else:
        skip = set(PYCHARM_SKIP or [])

    if only:
        skip = set(SWEEP_NAMES) - only

    os.makedirs(out_root, exist_ok=True)
    write_readme(out_root)

    if smoke:
        seeds = (0,)
        slots = 30
        users = min(users, 10)
        snr_targets = [0, 6, 10]
        user_list = [5, 15, 25]
        rates = [0.08, 0.16, 0.24]
        pick_snr = [2, 6]
        pick_users = [10, 20]
        betas = [0.55, 1.0]
    else:
        snr_targets = [0, 2, 4, 6, 8, 10]
        user_list = list(DEFAULT_USER_LIST)
        rates = list(DEFAULT_ARRIVAL_RATES)
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
        "causal_drift_gain=1.0; GDO=online emp Acc-floor; Acc table env-only"
    )

    if resume:
        for name in SWEEP_NAMES:
            if sweep_complete(out_root, name):
                skip.add(name)
                notes.append(f"resume: skip complete `{name}`")

    t0 = time.time()
    log(f"START sweeps smoke={smoke} algos={algos} skip={sorted(skip)} "
        f"cwd={os.getcwd()}")

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

    if not jobs:
        log("Nothing to run (all sweeps skipped).")
        return

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
    for sweep in SWEEP_NAMES:
        d = os.path.join(out_root, sweep)
        if os.path.isdir(d):
            arts = sorted(os.listdir(d))
            notes.append(f"{sweep} artifacts: {', '.join(arts)}")
    write_status(out_root, finished, failed, notes)
    log(f"DONE wall={elapsed/60:.1f} min finished={finished} failed={failed}")


if __name__ == "__main__":
    main()
