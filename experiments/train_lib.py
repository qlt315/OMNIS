"""Shared training / evaluation runner for OMNIS schemes.

Writes (and merges) CSVs + per-seed series under ``--out``.
Plotting is separate: ``experiments/plot_results.py``.

Reported **reward** = mean Lyapunov objective V·utility + drift.
Also logs accuracy, delay, energy, backlog, violation rate, and
per-slot wall time = decision + BCD + update.

**distributed decision_ms = parallel (max-agent)**: for factorized /
multi-agent algos (causal, ucb, dts, mappo, …) selection (+ local update)
is timed per MD and aggregated as max (or batched NN forward for MAPPO).
Centralized joint methods (cto, dqn, ppo) keep true sequential/joint wall.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np

from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.rss_main import RSS
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.dqn_main import DQN
from baselines.ppo_main import PPO
from baselines.mappo_main import MAPPO
from baselines.cto_main import CTO

try:
    from comm_model import comm_ms_per_slot
except ImportError:  # when imported as experiments.train_lib
    from experiments.comm_model import comm_ms_per_slot

ALGOS = [
    ("causal", OMNIS),
    ("ucb", OMNIS),
    ("dts", DTS),
    ("gdo", GDO),
    ("rss", RSS),
    ("dqn", DQN),
    ("ppo", PPO),
    ("mappo", MAPPO),
    ("cto", CTO),
]
ALGO_NAMES = [a for a, _ in ALGOS]

SERIES_KEYS = (
    "rew_series", "cum_reward", "acc_series", "delay_series",
    "energy_series", "backlog_series", "vio_series",
)
# Optional per-seed series (written when present on the agent / row).
OPTIONAL_SERIES_KEYS = (
    "pred_err_prior", "pred_err_post", "pred_err_reward", "pred_err_rmse",
)

# MAB family (optional pred-error logging default).
MAB_BASE_ALGOS = ("causal", "ucb", "dts", "cto")

# Factorized / multi-agent: decision_ms uses parallel (max-agent) timing.
DISTRIBUTED_DECISION_ALGOS = frozenset({
    "causal", "ucb", "dts", "mappo", "rss", "gdo",
})
# Centralized joint controllers: true sequential/joint decision wall.
CENTRALIZED_DECISION_ALGOS = frozenset({"cto", "dqn", "ppo"})

SCALAR_FIELDS = [
    "name", "seed", "reward", "acc", "delay", "energy", "backlog", "vio",
    "ms_per_slot", "decision_ms", "comm_ms", "bcd_ms", "update_ms",
    "comm_uplink_B", "comm_downlink_B", "comm_rounds", "sec",
]


def mean_series(agent, key):
    arrs = [np.asarray(agent.instant_metrics[u][key], dtype=float)
            for u in agent.users]
    return np.mean(np.stack(arrs, axis=0), axis=0)


def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def reward_series(agent):
    """Per-slot mean Lyapunov objective (reported reward)."""
    V = agent.lyapunov_v
    T = len(agent.instant_metrics[agent.users[0]]["reward"])
    obj = np.zeros(T)
    for t in range(T):
        slot = 0.0
        for u in agent.users:
            r = agent.instant_metrics[u]["reward"][t]
            energy = agent.instant_metrics[u]["energy"][t]
            backlog_t = agent.instant_metrics[u]["backlog"][t]
            eq_t = agent.instant_metrics[u]["energy_queue"][t]
            served_t = agent.instant_metrics[u]["served"][t]
            arrivals_t = agent.instant_metrics[u]["arrivals"][t]
            q_n = float(np.tanh(backlog_t / agent.dpp_bit_scale))
            z_n = float(np.tanh(eq_t / agent.dpp_energy_scale))
            service_n = served_t / agent.dpp_bit_scale
            arrivals_n = arrivals_t / agent.dpp_bit_scale
            budget_n = agent.energy_budget[u] / agent.dpp_energy_scale
            energy_n = energy / agent.dpp_energy_scale
            drift = q_n * (service_n - arrivals_n) + z_n * (budget_n - energy_n)
            slot += V * r + drift
        obj[t] = slot / len(agent.users)
    return obj


def algo_ms_per_slot(agent, slots, name=None):
    """Per-slot times [ms]: decision (+update), BCD, and modeled communication.

    For ``DISTRIBUTED_DECISION_ALGOS``, ``agent.decision_time`` /
    ``update_time`` already store parallel (max-agent) seconds — not the
    sum of sequential per-user simulator loops. Centralized algos keep
    joint/sequential wall.
    """
    dec = float(getattr(agent, "decision_time", 0.0))
    bcd = float(getattr(agent, "bcd_time", 0.0))
    upd = float(getattr(agent, "update_time", 0.0))
    # "decision" bar = agent compute (selection + learning update)
    # distributed decision_ms = parallel (max-agent)
    decision_ms = 1000.0 * (dec + upd) / max(slots, 1)
    bcd_ms = 1000.0 * bcd / max(slots, 1)
    local_dim = int(getattr(agent, "local_obs_dim", 4 + agent.top_l_cells + 2))
    comm = comm_ms_per_slot(
        name or getattr(agent, "name", "rss"),
        user_num=agent.user_num,
        local_obs_dim=local_dim,
        rtt_s=float(getattr(agent, "comm_rtt_s", 1e-3)),
        ctrl_rate_bps=float(getattr(agent, "comm_ctrl_rate_bps", 1e6)),
    )
    return {
        "decision_ms": decision_ms,
        "update_ms": 1000.0 * upd / max(slots, 1),  # kept for diagnostics
        "bcd_ms": bcd_ms,
        "comm_ms": float(comm["comm_ms"]),
        "comm_uplink_B": float(comm["comm_uplink_B"]),
        "comm_downlink_B": float(comm["comm_downlink_B"]),
        "comm_rounds": float(comm["comm_rounds"]),
        "ms_per_slot": decision_ms + bcd_ms + float(comm["comm_ms"]),
    }


def configure(name, seed, slots, users, *, no_update=False, freeze_after=0,
              log_pred_error=None):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    # CTO knobs (cto_max_candidates, cto_gp_burn_in) come from Config — keep
    # full joint GP cost; do not override to a light candidate pool here.
    if name in ("causal", "ucb"):
        c.algo = name
    if name == "dqn":
        c.dqn_eval = False
        c.rl_eval = False
        c.dqn_eps_decay_slots = max(1, slots)
    if name == "ppo":
        c.ppo_eval = False
        c.rl_eval = False
        c.ppo_rollout_len = min(16, max(8, slots // 4))
    if name == "mappo":
        c.mappo_eval = False
        c.rl_eval = False
        c.mappo_rollout_len = min(16, max(8, slots // 4))
    c.mab_no_update = bool(no_update)
    c.mab_freeze_after = int(freeze_after or 0)
    # Default: log pred error for MAB family (Causal acc + UCB/DTS/CTO reward).
    if log_pred_error is None:
        log_pred_error = name in MAB_BASE_ALGOS
    c.log_pred_error = bool(log_pred_error)
    return c


def _rolling_rmse(abs_errs):
    """Cumulative RMSE of absolute errors: sqrt(mean(e[:t+1]^2))."""
    e = np.asarray(abs_errs, dtype=float)
    if e.size == 0:
        return e
    return np.sqrt(np.cumsum(e * e) / np.arange(1, len(e) + 1))


def run_one(name, cls, seed, slots, users, *, base_algo=None,
            no_update=False, freeze_after=0, log_pred_error=None):
    """Run one (algo, seed). ``name`` is the result label; ``base_algo`` selects class/config."""
    algo = base_algo or name
    c = configure(algo, seed, slots, users, no_update=no_update,
                  freeze_after=freeze_after, log_pred_error=log_pred_error)
    t0 = time.time()
    agent = cls(c)
    # expose comm model knobs on agent for timing helper
    agent.comm_rtt_s = getattr(c, "comm_rtt_s", 1e-3)
    agent.comm_ctrl_rate_bps = getattr(c, "comm_ctrl_rate_bps", 1e6)
    agent.simulation()
    wall = time.time() - t0
    avg = agent.average_metrics
    obj = reward_series(agent)
    timing = algo_ms_per_slot(agent, slots, name=algo)
    row = {
        "name": name, "seed": seed,
        "reward": float(np.mean(obj)),
        "acc": float(avg["accuracy"]),
        "delay": float(avg["latency"]),
        "energy": float(avg["energy"]),
        "backlog": float(avg.get("backlog_bits", float("nan"))),
        "vio": float(avg["vio_prob"]),
        "rew_series": obj,
        "cum_reward": cumulative_mean(obj),
        "acc_series": mean_series(agent, "accuracy"),
        "delay_series": mean_series(agent, "delay"),
        "energy_series": mean_series(agent, "energy"),
        "backlog_series": mean_series(agent, "backlog"),
        "vio_series": mean_series(agent, "is_vio"),
        "sec": wall,
        **timing,
    }
    # Prediction-error series (Causal accuracy GP; optional reward GP for UCB-family)
    prior = getattr(agent, "pred_err_prior", None)
    post = getattr(agent, "pred_err_post", None)
    rew_err = getattr(agent, "pred_err_reward", None)
    if prior is not None and len(prior) > 0:
        row["pred_err_prior"] = np.asarray(prior, dtype=float)
        row["pred_err_rmse"] = _rolling_rmse(prior)
    if post is not None and len(post) > 0:
        row["pred_err_post"] = np.asarray(post, dtype=float)
        # Prefer posterior for rolling RMSE when available
        row["pred_err_rmse"] = _rolling_rmse(post)
    if rew_err is not None and len(rew_err) > 0:
        row["pred_err_reward"] = np.asarray(rew_err, dtype=float)
    return row


def series_dir(out_dir):
    return os.path.join(out_dir, "series")


def save_series(out_dir, row):
    """Persist per-seed time series for later plotting."""
    d = series_dir(out_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{row['name']}_seed{row['seed']}.npz")
    payload = {k: np.asarray(row[k], dtype=np.float64) for k in SERIES_KEYS}
    for k in OPTIONAL_SERIES_KEYS:
        if k in row and row[k] is not None:
            payload[k] = np.asarray(row[k], dtype=np.float64)
    payload["seed"] = np.asarray([row["seed"]], dtype=np.int64)
    np.savez_compressed(path, **payload)
    return path


def load_perseed_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_perseed_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCALAR_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in SCALAR_FIELDS:
                v = r[k]
                if isinstance(v, float):
                    out[k] = f"{v:.6g}"
                else:
                    out[k] = v
            w.writerow(out)


def rebuild_summary(out_dir, rows):
    """Write summary.csv from full per-seed rows (any subset of algos)."""
    agg = defaultdict(lambda: defaultdict(list))
    metrics = ["reward", "acc", "delay", "energy", "backlog", "vio",
               "ms_per_slot", "decision_ms", "comm_ms", "bcd_ms", "update_ms",
               "comm_uplink_B", "comm_downlink_B", "comm_rounds", "sec"]
    names = []
    for r in rows:
        name = r["name"]
        if name not in names:
            names.append(name)
        for m in metrics:
            if m in r:
                agg[name][m].append(float(r[m]))

    # Stable order: known ALGOS first, then any extras.
    ordered = [n for n in ALGO_NAMES if n in agg] + [
        n for n in names if n not in ALGO_NAMES]

    summary = os.path.join(out_dir, "summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "metric", "mean", "std"])
        for name in ordered:
            for m in metrics:
                v = np.asarray(agg[name][m], dtype=float)
                std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                w.writerow([name, m, f"{v.mean():.6g}", f"{std:.6g}"])
    return agg, ordered


def merge_results(out_dir, new_rows):
    """Merge new runs into perseed.csv (replace same name+seed), refresh summary."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "perseed.csv")
    existing = load_perseed_csv(path)
    replaced = {(r["name"], int(r["seed"])) for r in new_rows}
    kept = [r for r in existing
            if (r["name"], int(r["seed"])) not in replaced]
    # normalize types for kept rows
    merged = []
    for r in kept:
        merged.append({
            "name": r["name"], "seed": int(r["seed"]),
            **{k: float(r[k]) for k in SCALAR_FIELDS if k not in ("name", "seed")},
        })
    for r in new_rows:
        merged.append({k: r[k] for k in SCALAR_FIELDS})
        save_series(out_dir, r)

    # Sort: algo order then seed; unknown labels after known algos
    rank = {n: i for i, n in enumerate(ALGO_NAMES)}
    merged.sort(key=lambda r: (rank.get(r["name"], 999), r["name"], r["seed"]))
    write_perseed_csv(path, merged)
    agg, ordered = rebuild_summary(out_dir, merged)
    return merged, agg, ordered


def print_summary(agg, algo_names):
    print("\n=== SUMMARY (mean ± std) ===")
    hdr = (f"{'name':8} {'reward':>10} {'acc':>8} {'delay':>8} {'energy':>8} "
           f"{'backlog':>10} {'vio':>8} {'ms/slot':>10}")
    print(hdr)
    for name in algo_names:
        if name not in agg:
            continue

        def fmt(m, w=8):
            v = np.asarray(agg[name][m])
            s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            return f"{v.mean():{w}.3f}±{s:.2f}"

        print(f"{name:8} {fmt('reward',10)} {fmt('acc')} {fmt('delay')} "
              f"{fmt('energy')} {fmt('backlog',10)} {fmt('vio')} "
              f"{fmt('ms_per_slot',10)}")


def resolve_algos(names):
    """Validate and order algo names; raise on unknown."""
    if not names:
        return list(ALGO_NAMES)
    unknown = [n for n in names if n not in ALGO_NAMES]
    if unknown:
        raise SystemExit(
            f"unknown algos {unknown}; choose from {ALGO_NAMES}")
    # preserve user order, dedupe
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def run_training(algos, slots=300, users=10, seeds=(0, 1, 2, 3, 4),
                 out="figures/train", *, log_pred_error=None):
    """Run listed algos over seeds; merge CSVs + series under ``out`` (no plots)."""
    names = resolve_algos(algos)
    cls_map = dict(ALGOS)
    os.makedirs(out, exist_ok=True)

    rows = []
    for name in names:
        cls = cls_map[name]
        for seed in seeds:
            print(f"=== {name} seed={seed} ===", flush=True)
            r = run_one(name, cls, seed, slots, users,
                        log_pred_error=log_pred_error)
            print(f"    reward={r['reward']:.4f} acc={r['acc']:.4f} "
                  f"delay={r['delay']:.3f} energy={r['energy']:.3f} "
                  f"backlog={r['backlog']:.0f} vio={r['vio']:.3f} "
                  f"ms/slot={r['ms_per_slot']:.2f} "
                  f"(dec={r['decision_ms']:.2f} comm={r['comm_ms']:.2f} "
                  f"bcd={r['bcd_ms']:.2f}) wall={r['sec']:.0f}s",
                  flush=True)
            rows.append(r)

    _, agg, ordered = merge_results(out, rows)
    print_summary(agg, ordered)
    print(f"wrote/merged CSVs + series -> {out}/")
    print(f"plot with: PYTHONPATH=. python3 experiments/plot_results.py --indir {out}")
    return rows


def cli_main(default_algos=None):
    p = argparse.ArgumentParser(
        description="Train / evaluate OMNIS schemes (writes data only; use plot_results.py to plot)")
    p.add_argument(
        "--algos", nargs="+", default=default_algos,
        metavar="NAME",
        help=f"schemes to run (default: script-specific or all). Choices: {', '.join(ALGO_NAMES)}")
    # Overnight hard defaults (match sys_data/config.py stress scenario).
    p.add_argument("--slots", type=int, default=300)
    p.add_argument("--users", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="figures/train",
                   help="CSVs + series output dir (pair with plot_results --indir)")
    args = p.parse_args()
    algos = args.algos if args.algos is not None else list(ALGO_NAMES)
    run_training(algos, slots=args.slots, users=args.users,
                 seeds=tuple(args.seeds), out=args.out)
