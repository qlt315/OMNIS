"""Multi-seed long-horizon comparison under the Lyapunov DPP framework.

Runs all algorithms over N seeds and reports mean ± std of the key metrics,
plus writes per-seed series for convergence-curve plotting.

Usage:
  PYTHONPATH=. python scripts/run_multiseed.py --slots 200 --seeds 0 1 2 3 4
"""

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
from baselines.mappo_main import MAPPO
from baselines.cto_main import CTO

ALGOS = [
    ("causal", OMNIS),
    ("ucb", OMNIS),
    ("dts", DTS),
    ("gdo", GDO),
    ("rss", RSS),
    ("dqn", DQN),
    ("mappo", MAPPO),
    ("cto", CTO),  # last: ~3s/slot joint GP
]

LABELS = {
    "causal": "OMNIS-Causal", "ucb": "OMNIS-UCB",
    "dqn": "DQN (centralized)", "mappo": "MAPPO",
    "gdo": "GDO", "rss": "RSS", "dts": "OMNIS-TS", "cto": "CTO",
}
COLORS = {
    "causal": "#1f77b4", "ucb": "#ff7f0e", "dqn": "#2ca02c",
    "mappo": "#e377c2",
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
    ker = np.ones(w) / w
    return np.convolve(x, ker, mode="valid")


def dpp_objective_series(agent):
    """Reconstruct per-slot DPP using the same signs as ``dpp_drift``.

    Uses post-update queue metrics as a proxy (exact pre-update Q is not
    stored). Matches maximize q*(service-arrivals)+z*(budget-energy).
    """
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


def agent_ms_per_slot(agent, slots):
    """Mean decision+update wall time per slot [ms], excluding env/BCD."""
    dec = float(getattr(agent, "decision_time", 0.0))
    upd = float(getattr(agent, "update_time", 0.0))
    return 1000.0 * (dec + upd) / max(slots, 1)


def run_one(name, cls, seed, slots, users):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.cto_max_candidates = 2000
    if name in ("causal", "ucb"):
        c.algo = name
    if name == "dqn":
        # Online-from-scratch centralized branching DQN.
        c.dqn_eval = False
        c.rl_eval = False
        c.dqn_eps_decay_slots = max(1, slots)
    if name == "mappo":
        c.mappo_eval = False
        c.rl_eval = False
        c.mappo_rollout_len = min(32, max(8, slots // 4))
    t0 = time.time()
    agent = cls(c)
    agent.simulation()
    elapsed = time.time() - t0
    avg = agent.average_metrics
    dpp = dpp_objective_series(agent)
    ms_slot = agent_ms_per_slot(agent, slots)
    return {
        "name": name, "seed": seed,
        "reward": avg["reward"], "acc": avg["accuracy"], "lat": avg["latency"],
        "vio": avg["vio_prob"], "backlog": avg.get("backlog_bits", float("nan")),
        "cum_dpp": cumulative_mean(dpp), "dpp_series": dpp,
        "rew_series": mean_series(agent, "reward"),
        "acc_series": mean_series(agent, "accuracy"),
        "backlog_series": mean_series(agent, "backlog"),
        "cum_reward": cumulative_mean(mean_series(agent, "reward")),
        "sec": elapsed,
        "ms_per_slot": ms_slot,
        "decision_s": float(getattr(agent, "decision_time", 0.0)),
        "update_s": float(getattr(agent, "update_time", 0.0)),
    }


def bandplot(ax, series_list, color, label, sliding=None, lw=1.8, alpha=0.15):
    """Plot mean curve ± std band across seeds. No-op if series_list empty."""
    if not series_list:
        return
    L = min(len(s) for s in series_list)
    arr = np.stack([s[:L] for s in series_list], axis=0)  # [nseed, T]
    if sliding:
        arr = np.stack([sliding_mean(row, sliding) for row in arr], axis=0)
        x = np.arange(arr.shape[1]) + sliding - 1
    else:
        x = np.arange(arr.shape[1])
    m = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
    ax.plot(x, m, color=color, lw=lw, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=alpha)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=200)
    p.add_argument("--users", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="figures")
    p.add_argument("--slide", type=int, default=5,
                   help="sliding window for lightly-smoothed curves (default 5)")
    p.add_argument("--early", type=int, default=50,
                   help="early-horizon zoom length for learning curves")
    p.add_argument("--algos", nargs="+", default=None,
                   help="subset of algo names to run (default: all)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    algo_list = ALGOS
    if args.algos:
        want = set(args.algos)
        algo_list = [(n, c) for n, c in ALGOS if n in want]
        if not algo_list:
            raise SystemExit(f"no matching algos in {args.algos}")

    all_rows = []
    for name, cls in algo_list:
        for seed in args.seeds:
            print(f"=== {name} seed={seed} ===", flush=True)
            r = run_one(name, cls, seed, args.slots, args.users)
            print(f"    reward={r['reward']:.4f} vio={r['vio']:.4f} "
                  f"backlog={r['backlog']:.0f} "
                  f"ms/slot={r['ms_per_slot']:.2f} "
                  f"({r['sec']:.0f}s wall)", flush=True)
            all_rows.append(r)

    # --- per-seed CSV ---
    csv_path = os.path.join(args.out, "multiseed_perseed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "seed", "reward", "acc", "lat", "vio", "backlog",
                    "cum_dpp_final", "cum_reward_final", "final_backlog",
                    "sec", "ms_per_slot", "decision_s", "update_s"])
        for r in all_rows:
            w.writerow([r["name"], r["seed"], f"{r['reward']:.4f}", f"{r['acc']:.4f}",
                        f"{r['lat']:.4f}", f"{r['vio']:.4f}", f"{r['backlog']:.1f}",
                        f"{r['cum_dpp'][-1]:.4f}", f"{r['cum_reward'][-1]:.4f}",
                        f"{r['backlog_series'][-1]:.1f}", f"{r['sec']:.1f}",
                        f"{r['ms_per_slot']:.3f}",
                        f"{r['decision_s']:.3f}", f"{r['update_s']:.3f}"])

    # --- aggregate mean ± std table ---
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        for key in ["reward", "acc", "vio", "backlog", "sec", "ms_per_slot",
                    "decision_s", "update_s"]:
            agg[r["name"]][key].append(r[key])
        agg[r["name"]]["cum_dpp"].append(r["cum_dpp"][-1])
        agg[r["name"]]["cum_reward"].append(r["cum_reward"][-1])
        agg[r["name"]]["final_backlog"].append(r["backlog_series"][-1])

    ran_names = [n for n, _ in algo_list]

    agg_path = os.path.join(args.out, "multiseed_summary.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "metric", "mean", "std"])
        for name in ran_names:
            for metric in ["cum_dpp", "cum_reward", "reward", "acc", "vio",
                           "backlog", "final_backlog", "sec", "ms_per_slot",
                           "decision_s", "update_s"]:
                vals = np.array(agg[name][metric], dtype=float)
                w.writerow([name, metric, f"{vals.mean():.4f}",
                            f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}"])

    print("\n=== MULTI-SEED SUMMARY (mean ± std over seeds) ===")
    print(f"{'name':8} {'cumReward':>16} {'accuracy':>14} {'vio':>12} "
          f"{'ms/slot':>12} {'wall_s':>10}")
    for name in ran_names:
        cr = np.array(agg[name]["cum_reward"]); ac = np.array(agg[name]["acc"])
        vv = np.array(agg[name]["vio"]); ms = np.array(agg[name]["ms_per_slot"])
        rt = np.array(agg[name]["sec"])
        print(f"{name:8} {cr.mean():7.3f}±{cr.std(ddof=1):<7.3f} "
              f"{ac.mean():5.3f}±{ac.std(ddof=1):<6.3f} "
              f"{vv.mean():4.3f}±{vv.std(ddof=1):<5.3f} "
              f"{ms.mean():6.2f}±{ms.std(ddof=1):<5.2f} {rt.mean():9.1f}")

    def series_of(name, key):
        return [r[key] for r in all_rows if r["name"] == name]

    # --- cumulative gap vs GDO (myopic SF-ESP greedy baseline) ---
    # Positive ⇒ better reward than GDO; not a mathematical regret bound.
    gdo_by_seed = {r["seed"]: r["rew_series"] for r in all_rows if r["name"] == "gdo"}
    gap = {}
    for name in ran_names:
        if name == "gdo":
            continue
        gap[name] = []
        for r in all_rows:
            if r["name"] != name:
                continue
            g = gdo_by_seed.get(r["seed"])
            if g is None:
                continue
            L = min(len(g), len(r["rew_series"]))
            inst = np.asarray(r["rew_series"][:L]) - np.asarray(g[:L])
            gap[name].append(np.cumsum(inst))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        if name == "gdo" or name not in gap or not gap[name]:
            continue
        bandplot(ax, gap[name], COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative reward gap vs GDO")
    ax.set_title("MAIN: cumulative reward advantage over GDO")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_cum_regret.png"), dpi=160); plt.close(fig)

    # early-horizon zoom
    early = min(args.early, args.slots)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        if name == "gdo" or name not in gap or not gap[name]:
            continue
        bandplot(ax, [s[:early] for s in gap[name]], COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative reward gap vs GDO")
    ax.set_title(f"Early gap vs GDO (first {early} slots)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_cum_regret_early.png"), dpi=160); plt.close(fig)

    # backlog
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, series_of(name, "backlog_series"), COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Mean backlog (bits)")
    ax.set_title("Queue stability: mean ± std over seeds")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_backlog.png"), dpi=160); plt.close(fig)

    # accuracy (raw-ish: light slide)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, series_of(name, "acc_series"), COLORS[name], LABELS[name],
                 sliding=args.slide)
    ax.set_xlabel("Time slot"); ax.set_ylabel("Mean accuracy (mAP)")
    ax.set_title(f"Task accuracy (sliding W={args.slide}): mean ± std")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_accuracy.png"), dpi=160); plt.close(fig)

    # accuracy learning (cumulative mean)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, [cumulative_mean(s) for s in series_of(name, "acc_series")],
                 COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative mean accuracy")
    ax.set_title("Accuracy learning curve: mean ± std")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_acc_learning.png"), dpi=160); plt.close(fig)

    # early accuracy learning zoom
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, [cumulative_mean(s)[:early] for s in series_of(name, "acc_series")],
                 COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative mean accuracy")
    ax.set_title(f"Early accuracy learning (first {early} slots)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_acc_learning_early.png"), dpi=160); plt.close(fig)

    # cumulative DPP
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, series_of(name, "cum_dpp"), COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative mean DPP objective")
    ax.set_title("Cumulative DPP objective: mean ± std over seeds")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_cum_dpp.png"), dpi=160); plt.close(fig)

    # lightly-smoothed DPP / reward (W=5 default; was W=20 and too smooth)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, series_of(name, "dpp_series"), COLORS[name], LABELS[name],
                 sliding=args.slide)
    ax.set_xlabel("Time slot"); ax.set_ylabel(f"DPP objective (sliding W={args.slide})")
    ax.set_title("Per-slot DPP objective (light smooth): mean ± std")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_sliding_dpp.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, series_of(name, "rew_series"), COLORS[name], LABELS[name],
                 sliding=args.slide)
    ax.set_xlabel("Time slot"); ax.set_ylabel(f"Mean reward (sliding W={args.slide})")
    ax.set_title("Per-slot reward (light smooth): mean ± std")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_sliding_reward.png"), dpi=160); plt.close(fig)

    # cumulative mean reward (shows slow MAB improvement better than sliding)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, series_of(name, "cum_reward"), COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative mean reward")
    ax.set_title("Reward learning (cumulative mean): mean ± std")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_cum_reward.png"), dpi=160); plt.close(fig)

    # early cumulative mean reward
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ran_names:
        bandplot(ax, [s[:early] for s in series_of(name, "cum_reward")],
                 COLORS[name], LABELS[name])
    ax.set_xlabel("Time slot"); ax.set_ylabel("Cumulative mean reward")
    ax.set_title(f"Early reward learning (first {early} slots)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "ms_cum_reward_early.png"), dpi=160); plt.close(fig)

    # --- Agent runtime figures -------------------------------------------------
    # Numbers are correct but easy to misread:
    #   UCB/TS  : sklearn GP refit (L-BFGS) every slot  -> hundreds of ms
    #   Causal  : shared residual GP + rank-1 Cholesky  -> tens of ms
    #   GDO     : SF-ESP greedy (semantic z* + EG)      -> tens of ms
    #   DQN     : centralized branching Q-net + SGD     -> few–tens ms
    #   MAPPO   : shared actor + central critic (PPO)   -> few–tens ms
    #   RSS     : static random arms, no online work    -> ~0
    # Main paper claim is Causal vs UCB/TS; plot that family first.
    slots = args.slots
    # GP-family baselines (fair complexity class for the runtime claim)
    mab_names = [n for n in ["causal", "ucb", "dts", "cto"] if n in ran_names]
    all_names = list(ran_names)

    def _ms(name, key):
        # key in {"ms_per_slot"} or reconstruct from decision_s/update_s
        return float(np.mean(agg[name][key]))

    def _ms_std(name, key):
        vals = np.asarray(agg[name][key], dtype=float)
        return float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    # (1) MAIN: GP-MAB family only (fair complexity class)
    if mab_names:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        x = np.arange(len(mab_names))
        means = [_ms(n, "ms_per_slot") for n in mab_names]
        stds = [_ms_std(n, "ms_per_slot") for n in mab_names]
        ax.bar(x, means, yerr=stds, color=[COLORS[n] for n in mab_names],
               capsize=4, alpha=0.9, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[n] for n in mab_names])
        ax.set_ylabel("Agent time per slot [ms]")
        ax.set_title("MAIN: GP-MAB decision+update time / slot\n(excl. channel estimation & BCD)")
        ax.grid(True, axis="y", alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + max(means) * 0.03, f"{m:.1f}", ha="center", fontsize=9)
        if "ucb" in mab_names and "causal" in mab_names:
            ucb_m = means[mab_names.index("ucb")]
            cau_m = means[mab_names.index("causal")]
            ax.annotate(f"{ucb_m / max(cau_m, 1e-9):.1f}× faster",
                        xy=(mab_names.index("causal"), cau_m),
                        xytext=(0.35, ucb_m * 0.55),
                        fontsize=10, color=COLORS["causal"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["causal"], lw=1.2))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ms_runtime.png"), dpi=160)
        plt.close(fig)

        # stacked decision vs update for MAB family
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        dec_m = [np.mean(agg[n]["decision_s"]) * 1000.0 / slots for n in mab_names]
        upd_m = [np.mean(agg[n]["update_s"]) * 1000.0 / slots for n in mab_names]
        ax.bar(x, dec_m, color=[COLORS[n] for n in mab_names], alpha=0.9,
               label="decision (suggest)")
        ax.bar(x, upd_m, bottom=dec_m, color=[COLORS[n] for n in mab_names],
               alpha=0.4, label="update (register/fit)")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[n] for n in mab_names])
        ax.set_ylabel("Time per slot [ms]")
        ax.set_title("GP-MAB: decision vs update breakdown")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ms_runtime_stack.png"), dpi=160)
        plt.close(fig)

    # (2) All schemes, log scale + category notes (appendix / fairness check)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x_all = np.arange(len(all_names))
    means_all = [_ms(n, "ms_per_slot") for n in all_names]
    stds_all = [_ms_std(n, "ms_per_slot") for n in all_names]
    # floor for log display (RSS ~0)
    plot_means = [max(m, 1e-3) for m in means_all]
    ax.bar(x_all, plot_means, color=[COLORS[n] for n in all_names], alpha=0.9, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x_all)
    ax.set_xticklabels([LABELS[n] for n in all_names], rotation=18, ha="right")
    ax.set_ylabel("Agent time per slot [ms] (log)")
    ax.set_title("All schemes (log scale)\n"
                 "UCB/TS/CTO: GP+L-BFGS · Causal: Cholesky GP · "
                 "GDO: SF-ESP greedy · DQN/MAPPO: NN · RSS: static")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    notes = {
        "rss": "no learning",
        "dqn": "central Q",
        "mappo": "CTDE PPO",
        "gdo": "SF-ESP greedy",
        "causal": "shared GP",
        "ucb": "per-MD GP",
        "dts": "per-MD GP",
        "cto": "joint GP",
    }
    for i, n in enumerate(all_names):
        ax.text(i, plot_means[i] * 1.35,
                f"{means_all[i]:.2f}\n({notes.get(n, '')})",
                ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ms_runtime_all_log.png"), dpi=160)
    plt.close(fig)

    print(f"\nwrote multi-seed plots + csv to {args.out}/")
    print("Runtime note: UCB/TS cost is GP refit (L-BFGS); GDO/DQN/RSS have no GP, "
          "so lower ms/slot is expected — see ms_runtime.png (MAB-only) vs "
          "ms_runtime_all_log.png.")


if __name__ == "__main__":
    main()
