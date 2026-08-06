"""Shared training / evaluation runner for OMNIS schemes.

Reported **reward** = mean Lyapunov objective V·utility + drift.
Also logs accuracy, delay, energy, backlog, violation rate, and
per-slot wall time = decision + BCD + update.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
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

LABELS = {
    "causal": "OMNIS-Causal", "ucb": "OMNIS-UCB",
    "dqn": "DQN", "ppo": "PPO", "mappo": "MAPPO",
    "gdo": "GDO", "rss": "RSS", "dts": "OMNIS-TS", "cto": "CTO",
}
COLORS = {
    "causal": "#1f77b4", "ucb": "#ff7f0e", "dqn": "#2ca02c",
    "ppo": "#bcbd22", "mappo": "#e377c2",
    "gdo": "#d62728", "rss": "#9467bd", "dts": "#8c564b", "cto": "#17becf",
}


def mean_series(agent, key):
    arrs = [np.asarray(agent.instant_metrics[u][key], dtype=float)
            for u in agent.users]
    return np.mean(np.stack(arrs, axis=0), axis=0)


def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def sliding_mean(x, w=5):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="valid")


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


def algo_ms_per_slot(agent, slots):
    """Total algorithm time per slot [ms]: decision + BCD + update."""
    dec = float(getattr(agent, "decision_time", 0.0))
    bcd = float(getattr(agent, "bcd_time", 0.0))
    upd = float(getattr(agent, "update_time", 0.0))
    return {
        "ms_per_slot": 1000.0 * (dec + bcd + upd) / max(slots, 1),
        "decision_ms": 1000.0 * dec / max(slots, 1),
        "bcd_ms": 1000.0 * bcd / max(slots, 1),
        "update_ms": 1000.0 * upd / max(slots, 1),
    }


def configure(name, seed, slots, users):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.cto_max_candidates = 2000
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
    return c


def run_one(name, cls, seed, slots, users):
    c = configure(name, seed, slots, users)
    t0 = time.time()
    agent = cls(c)
    agent.simulation()
    wall = time.time() - t0
    avg = agent.average_metrics
    obj = reward_series(agent)
    timing = algo_ms_per_slot(agent, slots)
    return {
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


def bandplot(ax, series_list, color, label, sliding=None, lw=1.8, alpha=0.15):
    if not series_list:
        return
    L = min(len(s) for s in series_list)
    arr = np.stack([s[:L] for s in series_list], axis=0)
    if sliding:
        arr = np.stack([sliding_mean(row, sliding) for row in arr], axis=0)
        x = np.arange(arr.shape[1]) + sliding - 1
    else:
        x = np.arange(arr.shape[1])
    m, sd = arr.mean(axis=0), (arr.std(axis=0, ddof=1) if arr.shape[0] > 1
                               else np.zeros(arr.shape[1]))
    ax.plot(x, m, color=color, lw=lw, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=alpha)


def write_csvs(rows, out_dir, algo_names):
    os.makedirs(out_dir, exist_ok=True)
    fields = [
        "name", "seed", "reward", "acc", "delay", "energy", "backlog", "vio",
        "ms_per_slot", "decision_ms", "bcd_ms", "update_ms", "sec",
    ]
    perseed = os.path.join(out_dir, "perseed.csv")
    with open(perseed, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])
                        for k in fields})

    agg = defaultdict(lambda: defaultdict(list))
    metrics = ["reward", "acc", "delay", "energy", "backlog", "vio",
               "ms_per_slot", "decision_ms", "bcd_ms", "update_ms", "sec"]
    for r in rows:
        for m in metrics:
            agg[r["name"]][m].append(float(r[m]))

    summary = os.path.join(out_dir, "summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "metric", "mean", "std"])
        for name in algo_names:
            if name not in agg:
                continue
            for m in metrics:
                v = np.asarray(agg[name][m])
                std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                w.writerow([name, m, f"{v.mean():.6g}", f"{std:.6g}"])
    return agg


def plot_results(rows, algo_names, out_dir, slide=5):
    os.makedirs(out_dir, exist_ok=True)

    def series_of(name, key):
        return [r[key] for r in rows if r["name"] == name]

    # Reward
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in algo_names:
        bandplot(ax, series_of(name, "cum_reward"), COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative mean reward")
    ax.set_title("Reward (Lyapunov objective)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in algo_names:
        bandplot(ax, series_of(name, "rew_series"), COLORS[name], LABELS[name],
                 sliding=slide)
    ax.set_xlabel("Time slot"); ax.set_ylabel(f"Mean reward (W={slide})")
    ax.set_title("Per-slot reward")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_sliding.png"), dpi=160); plt.close(fig)

    # Components
    components = [
        ("acc_series", "accuracy.png", "Mean accuracy (mAP)", "Accuracy"),
        ("delay_series", "delay.png", "Mean delay [s]", "Delay"),
        ("energy_series", "energy.png", "Mean energy [J]", "Energy"),
        ("backlog_series", "backlog.png", "Mean backlog [bits]", "Queue backlog"),
        ("vio_series", "violation.png", "Violation rate", "Constraint violation"),
    ]
    for key, fname, ylabel, title in components:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for name in algo_names:
            bandplot(ax, series_of(name, key), COLORS[name], LABELS[name],
                     sliding=slide if key != "backlog_series" else None)
        ax.set_xlabel("Time slot"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160); plt.close(fig)

    # Runtime: stacked decision / BCD / update
    names = [n for n in algo_names if any(r["name"] == n for r in rows)]
    if not names:
        return
    dec = [np.mean([r["decision_ms"] for r in rows if r["name"] == n]) for n in names]
    bcd = [np.mean([r["bcd_ms"] for r in rows if r["name"] == n]) for n in names]
    upd = [np.mean([r["update_ms"] for r in rows if r["name"] == n]) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x, dec, label="decision", color="#4c78a8")
    ax.bar(x, bcd, bottom=dec, label="BCD", color="#f58518")
    bottom2 = [a + b for a, b in zip(dec, bcd)]
    ax.bar(x, upd, bottom=bottom2, label="update", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], rotation=18, ha="right")
    ax.set_ylabel("Wall time per slot [ms]")
    ax.set_title("Runtime per slot (decision + BCD + update)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "runtime.png"), dpi=160); plt.close(fig)

    # Log-scale total for readability across decades
    totals = [d + b + u for d, b, u in zip(dec, bcd, upd)]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    plot_t = [max(t, 1e-3) for t in totals]
    ax.bar(x, plot_t, color=[COLORS[n] for n in names], alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], rotation=18, ha="right")
    ax.set_ylabel("Total algo time / slot [ms] (log)")
    ax.set_title("Runtime per slot (log scale)")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "runtime_log.png"), dpi=160); plt.close(fig)


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


def run_training(algos, slots=200, users=6, seeds=(0, 1, 2, 3, 4),
                 out="figures", slide=5):
    """Run listed algos over seeds; write CSV + plots under ``out``."""
    want = set(algos)
    algo_list = [(n, c) for n, c in ALGOS if n in want]
    if not algo_list:
        raise SystemExit(f"unknown algos {algos}; choose from {[a for a, _ in ALGOS]}")
    names = [n for n, _ in algo_list]
    os.makedirs(out, exist_ok=True)

    rows = []
    for name, cls in algo_list:
        for seed in seeds:
            print(f"=== {name} seed={seed} ===", flush=True)
            r = run_one(name, cls, seed, slots, users)
            print(f"    reward={r['reward']:.4f} acc={r['acc']:.4f} "
                  f"delay={r['delay']:.3f} energy={r['energy']:.3f} "
                  f"backlog={r['backlog']:.0f} vio={r['vio']:.3f} "
                  f"ms/slot={r['ms_per_slot']:.2f} "
                  f"(dec={r['decision_ms']:.2f} bcd={r['bcd_ms']:.2f} "
                  f"upd={r['update_ms']:.2f}) wall={r['sec']:.0f}s",
                  flush=True)
            rows.append(r)

    agg = write_csvs(rows, out, names)
    plot_results(rows, names, out, slide=slide)
    print_summary(agg, names)
    print(f"wrote plots/CSVs -> {out}/")
    return rows


def cli_main(default_algos=None):
    p = argparse.ArgumentParser(description="Train / evaluate OMNIS schemes")
    p.add_argument("--algos", nargs="+", default=default_algos,
                   help="scheme names (default: all or script-specific)")
    p.add_argument("--slots", type=int, default=200)
    p.add_argument("--users", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="figures")
    p.add_argument("--slide", type=int, default=5)
    args = p.parse_args()
    algos = args.algos or [a for a, _ in ALGOS]
    run_training(algos, slots=args.slots, users=args.users,
                 seeds=tuple(args.seeds), out=args.out, slide=args.slide)
