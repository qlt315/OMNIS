"""Multi-scheme comparison under the Lyapunov DPP framework.

All algorithms (arm selection AND MCS selection) now optimize the same DPP
objective: V*reward + drift. Main metrics are therefore queue stability and
the cumulative DPP objective, with raw reward as secondary.

Runs causal / ucb / dqn (online) / gdo (SF-ESP greedy)
/ rss / dts and writes:
  figures/cum_dpp.png             — MAIN: cumulative DPP objective
  figures/backlog.png             — MAIN: queue backlog (stability) over time
  figures/cum_reward.png          — secondary: cumulative mean reward
  figures/sliding_reward.png      — sliding-window mean reward
  figures/vio_backlog.png         — violation prob + backlog over time
  figures/summary.csv             — numeric summary

Usage:
  PYTHONPATH=. python scripts/plot_compare.py
  PYTHONPATH=. python scripts/plot_compare.py --slots 80 --users 6
"""

import argparse
import csv
import os
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.rss_main import RSS
from baselines.dts_main import DTS
from baselines.gdo_main import GDO
from baselines.dqn_main import DQN
from baselines.mappo_main import MAPPO
from baselines.cto_main import CTO


def mean_series(agent, key):
    """Average a per-user instant metric across users → length T array."""
    arrs = [np.asarray(agent.instant_metrics[u][key], dtype=float)
            for u in agent.users]
    return np.mean(np.stack(arrs, axis=0), axis=0)


def cumulative_mean_reward(agent):
    r = mean_series(agent, "reward")
    return np.cumsum(r) / np.arange(1, len(r) + 1)


def dpp_objective_series(agent):
    """Per-slot DPP objective actually optimized: V*mean_reward + mean_drift.

    Reconstructs the drift term from queue/backlog state and each user's chosen
    (model, mcs). This is the true Lyapunov drift-plus-penalty objective that
    all algorithms now optimize (arm selection AND mcs selection are DPP-aware)."""
    V = agent.lyapunov_v
    T = len(agent.instant_metrics[agent.users[0]]["reward"])
    obj = np.zeros(T)
    for t in range(T):
        slot = 0.0
        for u in agent.users:
            r = agent.instant_metrics[u]["reward"][t]
            mcs = agent.instant_metrics[u]["mcs"][t]
            model = agent.instant_metrics[u]["model"][t] if "model" in agent.instant_metrics[u] else None
            # reconstruct energy & drift at slot t from recorded overhead
            energy = agent.instant_metrics[u]["energy"][t] if "energy" in agent.instant_metrics[u] else 0.0
            # approximate drift using current recorded backlog/energy_queue at t
            backlog_t = agent.instant_metrics[u]["backlog"][t]
            eq_t = agent.instant_metrics[u]["energy_queue"][t]
            q_n = backlog_t / agent.dpp_bit_scale
            z_n = eq_t / agent.dpp_energy_scale
            served_t = agent.instant_metrics[u]["served"][t] if "served" in agent.instant_metrics[u] else 0.0
            drift = (-q_n * served_t / agent.dpp_bit_scale
                     + z_n * (energy - agent.energy_budget[u]) / agent.dpp_energy_scale)
            slot += V * r + drift
        obj[t] = slot / len(agent.users)
    return obj


def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def sliding_mean(x, w=10):
    if len(x) < w:
        return x.copy()
    ker = np.ones(w) / w
    return np.convolve(x, ker, mode="valid")


def run_one(name, cls, seed, slots, users):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.cto_max_candidates = 2000
    if name in ("causal", "ucb"):
        c.algo = name
    t0 = time.time()
    if name == "dqn":
        c.dqn_eval = False
        c.rl_eval = False
        c.dqn_eps_decay_slots = max(1, slots)
    if name == "mappo":
        c.mappo_eval = False
        c.rl_eval = False
        c.mappo_rollout_len = min(32, max(8, slots // 4))
    agent = cls(c)
    agent.simulation()
    elapsed = time.time() - t0
    avg = agent.average_metrics
    cells = []
    for u in agent.users:
        cells.extend(agent.instant_metrics[u]["cell"])
    return {
        "name": name,
        "agent": agent,
        "reward": avg["reward"],
        "acc": avg["accuracy"],
        "lat": avg["latency"],
        "vio": avg["vio_prob"],
        "backlog": avg.get("backlog_bits", float("nan")),
        "cells": dict(sorted(Counter(cells).items())),
        "sec": elapsed,
        "cum_reward": cumulative_mean_reward(agent),
        "cum_dpp": cumulative_mean(dpp_objective_series(agent)),
        "dpp_series": dpp_objective_series(agent),
        "rew_series": mean_series(agent, "reward"),
        "vio_series": mean_series(agent, "is_vio"),
        "backlog_series": mean_series(agent, "backlog"),
    }


def plot_all(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = {
        "causal": "#1f77b4", "ucb": "#ff7f0e", "dqn": "#2ca02c",
        "mappo": "#e377c2",
        "gdo": "#d62728", "rss": "#9467bd", "dts": "#8c564b", "cto": "#17becf",
    }
    labels = {
        "causal": "OMNIS-Causal", "ucb": "OMNIS-UCB",
        "dqn": "DQN (centralized)", "mappo": "MAPPO",
        "gdo": "GDO", "rss": "RSS", "dts": "OMNIS-TS", "cto": "CTO",
    }

    # --- Main figure: cumulative DPP objective (the true Lyapunov objective) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in rows:
        ax.plot(r["cum_dpp"], label=labels.get(r["name"], r["name"]),
                color=colors.get(r["name"]), lw=2)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Cumulative mean DPP objective")
    ax.set_title("Main: cumulative Lyapunov drift-plus-penalty objective")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cum_dpp.png"), dpi=160)
    plt.close(fig)

    # --- Backlog stability (queue length over time) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in rows:
        ax.plot(r["backlog_series"], label=labels.get(r["name"], r["name"]),
                color=colors.get(r["name"]), lw=1.8)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Mean backlog (bits)")
    ax.set_title("Queue stability: mean backlog over time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "backlog.png"), dpi=160)
    plt.close(fig)

    # --- Cumulative mean reward (secondary) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in rows:
        ax.plot(r["cum_reward"], label=labels.get(r["name"], r["name"]),
                color=colors.get(r["name"]), lw=2)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Cumulative mean reward")
    ax.set_title("Secondary: cumulative mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cum_reward.png"), dpi=160)
    plt.close(fig)

    # --- Sliding reward ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in rows:
        y = sliding_mean(r["rew_series"], w=10)
        ax.plot(np.arange(len(y)) + 9, y, label=labels.get(r["name"], r["name"]),
                color=colors.get(r["name"]), lw=1.8)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Sliding-window mean reward (W=10)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sliding_reward.png"), dpi=160)
    plt.close(fig)

    # --- DQN training convergence curve (if available) ---
    train_csv = os.path.join(out_dir, "dqn_train_curve.csv")
    if os.path.exists(train_csv):
        ts, rs = [], []
        with open(train_csv) as f:
            for row in csv.DictReader(f):
                ts.append(int(row["slot"]))
                rs.append(float(row["mean_reward"]))
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(ts, rs, alpha=0.25, color="#2ca02c", lw=0.8)
        y = sliding_mean(np.array(rs), w=20)
        ax.plot(np.arange(len(y)) + 19, y, color="#2ca02c", lw=2,
                label="DQN training (sliding W=20)")
        ax.set_xlabel("Training slot")
        ax.set_ylabel("Mean reward")
        ax.set_title("DQN pretraining convergence")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dqn_train_curve.png"), dpi=160)
        plt.close(fig)

    # --- Vio + backlog ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in rows:
        axes[0].plot(sliding_mean(r["vio_series"], 10),
                     label=labels.get(r["name"], r["name"]),
                     color=colors.get(r["name"]), lw=1.6)
        axes[1].plot(sliding_mean(r["backlog_series"], 10),
                     label=labels.get(r["name"], r["name"]),
                     color=colors.get(r["name"]), lw=1.6)
    axes[0].set_title("Violation rate (sliding)")
    axes[0].set_xlabel("Time slot")
    axes[0].set_ylabel("P(violation)")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Backlog (sliding)")
    axes[1].set_xlabel("Time slot")
    axes[1].set_ylabel("Bits")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vio_backlog.png"), dpi=160)
    plt.close(fig)

    # --- CSV summary ---
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "reward", "acc", "latency", "vio", "backlog",
                    "cum_dpp_final", "cum_reward_final", "final_backlog", "sec"])
        for r in rows:
            cd = r["cum_dpp"]
            cr = r["cum_reward"]
            w.writerow([r["name"], f"{r['reward']:.4f}", f"{r['acc']:.4f}",
                        f"{r['lat']:.4f}", f"{r['vio']:.4f}", f"{r['backlog']:.1f}",
                        f"{cd[-1]:.4f}", f"{cr[-1]:.4f}",
                        f"{r['backlog_series'][-1]:.1f}", f"{r['sec']:.1f}"])
    print(f"wrote plots to {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=80)
    p.add_argument("--users", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    jobs = [
        ("causal", OMNIS),
        ("ucb", OMNIS),
        ("dts", DTS),
        ("cto", CTO),
        ("dqn", DQN),
        ("mappo", MAPPO),
        ("gdo", GDO),
        ("rss", RSS),
    ]
    rows = []
    for name, cls in jobs:
        print(f"=== running {name} ===", flush=True)
        r = run_one(name, cls, args.seed, args.slots, args.users)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in r.items() if k not in (
                   "agent", "cum_reward", "cum_dpp", "dpp_series", "rew_series",
                   "vio_series", "backlog_series")},
              flush=True)
        rows.append(r)

    print("\n=== SUMMARY (main metrics = cumulative DPP objective + queue backlog) ===")
    print(f"{'name':8} {'cumDPP':>8} {'cumR':>8} {'vio':>7} {'backlog':>9} {'fin_bkl':>9} {'sec':>6}")
    for r in rows:
        print(f"{r['name']:8} {r['cum_dpp'][-1]:8.4f} {r['cum_reward'][-1]:8.4f} "
              f"{r['vio']:7.4f} {r['backlog']:9.0f} {r['backlog_series'][-1]:9.0f} {r['sec']:6.1f}")

    plot_all(rows, args.out)


if __name__ == "__main__":
    main()
