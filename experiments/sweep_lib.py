"""Shared helpers for parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.convergence_lib import (
    ALGOS,
    ALGO_NAMES,
    algo_ms_per_slot,
    mean_task_reward,
    resolve_algos,
)
from experiments.result_io import load_mat_extras, save_mat_and_python
from sys_data.config import Config

SWEEP_UE_POOL_SIZE = 25
DEFAULT_SLOTS = 500
DEFAULT_USERS = 10
DEFAULT_USER_LIST = (5, 10, 15, 20, 25)
DEFAULT_ARRIVAL_RATES = (0.05, 0.10, 0.18, 0.28, 0.40)
DEFAULT_SWEEP_ALGOS = tuple(ALGO_NAMES)
METRICS = ("reward", "delay", "energy", "acc", "vio", "backlog")
LOG_Y_METRICS = frozenset({"delay", "backlog"})
METRIC_YLABEL = {
    "reward": "Avg. Reward (V·u+drift)",
    "delay": "Avg. Latency [s]",
    "energy": "Avg. Energy [J]",
    "acc": "Avg. Acc.",
    "vio": "Avg. Violation Prob. (delay|energy)",
    "backlog": "Backlog [tasks]",
}
LABELS = {
    "causal": "OMNIS-Causal", "ucb": "OMNIS-UCB",
    "dqn": "DQN", "ppo": "PPO", "mappo": "MAPPO",
    "gdo": "GDO", "dts": "OMNIS-TS", "cto": "C-OMNIS+",
}
COLORS = {
    "causal": "#1f77b4", "ucb": "#ff7f0e", "dqn": "#2ca02c",
    "ppo": "#bcbd22", "mappo": "#e377c2",
    "gdo": "#d62728", "dts": "#8c564b", "cto": "#17becf",
}
MODEL_LABELS = {
    "Box3": "Box-3", "Box6": "Box-6", "Box12": "Box-12",
    "Standard3": "Standard-3", "Standard6": "Standard-6",
    "Standard12": "Standard-12",
}


def _mean_best_cell_sinr_db(ue_ids, tag="smoke7", table_dir="phy_sim/output"):
    from omnis.sinr_trace import SinrTrace
    tr = SinrTrace.from_config_dir(table_dir, tag=tag, ue_ids=list(ue_ids))
    return float(np.mean(np.max(tr.sinr_db, axis=1)))


def fixed_ue_pool(pool_size=SWEEP_UE_POOL_SIZE, tag="smoke7",
                  table_dir="phy_sim/output"):
    from omnis.sinr_trace import SinrTrace, select_spread_ue_ids
    probe = SinrTrace.from_config_dir(table_dir, tag=tag)
    return select_spread_ue_ids(int(pool_size), probe.num_ues, probe.num_cells)


_UE_POOL_25 = fixed_ue_pool(SWEEP_UE_POOL_SIZE)
TRACE_MEAN_BEST_CELL_SINR_DB = _mean_best_cell_sinr_db(_UE_POOL_25)


def offset_for_snr_target(snr_target_db: float, users=None) -> float:
    """dB shift so mean best-cell trace SINR ≈ ``snr_target_db``."""
    n = int(users) if users is not None else SWEEP_UE_POOL_SIZE
    n = max(1, min(n, SWEEP_UE_POOL_SIZE))
    mean_db = (_mean_best_cell_sinr_db(_UE_POOL_25[:n])
               if n < SWEEP_UE_POOL_SIZE else TRACE_MEAN_BEST_CELL_SINR_DB)
    return float(snr_target_db) - mean_db


def configure_sweep(
    name, seed, slots, users, *,
    sinr_offset_db=0.0, arrival_rate=None, arrival_scale=None,
    causal_beta=None, beta_const=None, dqn_eps_end=None,
    gdo_feas_margin=None, reward_qos_coef=None, energy_hi=None,
    delay_hi=None, causal_feas_margin=None,
):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.sinr_offset_db = float(sinr_offset_db)

    _md = {
        "freq": 1.6, "cores": 768, "flops_per_cycle": 12,
        "power_coeff": 0.35, "trans_power": 0.1,
    }
    c.md_params = {u: dict(_md) for u in c.users}
    c.md_params_origin = {u: dict(_md) for u in c.users}
    c.es_params = {
        "freq": 4.0, "cores": 12288, "flops_per_cycle": 24,
        "power_coeff": 0.75,
    }
    c.es_params_origin = dict(c.es_params)

    if arrival_rate is not None:
        rate = float(arrival_rate)
        c.arrival_rate = {u: rate for u in c.users}
    elif arrival_scale is not None:
        c.arrival_rate = {u: 0.07 * float(arrival_scale) for u in c.users}
    else:
        c.arrival_rate = {u: 0.07 for u in c.users}
    c.arrival_rate_origin = dict(c.arrival_rate)

    if energy_hi is not None:
        lo = float(c.energy_constraint_range[0])
        c.energy_constraint_range = (lo, float(energy_hi))
        c.fixed_energy = {
            u: np.random.uniform(*c.energy_constraint_range) for u in c.users}
        c.fixed_energy_origin = dict(c.fixed_energy)

    if delay_hi is not None:
        lo = float(c.delay_constraint_range[0])
        c.delay_constraint_range = (lo, float(delay_hi))
        c.fixed_delay = {
            u: np.random.uniform(*c.delay_constraint_range) for u in c.users}
        c.fixed_delay_origin = dict(c.fixed_delay)

    if reward_qos_coef is not None:
        c.reward_qos_coef = float(reward_qos_coef)
    if causal_feas_margin is not None:
        c.causal_feas_margin = float(causal_feas_margin)
    if gdo_feas_margin is not None:
        c.gdo_feas_margin = float(gdo_feas_margin)

    if name in ("causal", "ucb"):
        c.algo = name
    if causal_beta is not None:
        c.causal_beta = float(causal_beta)
    if beta_const is not None:
        c.beta_const_val = float(beta_const)
        c.utility.beta_const = float(beta_const)

    if name == "dqn":
        c.dqn_eval = False
        c.rl_eval = False
        c.dqn_eps_decay_slots = max(1, slots)
        if dqn_eps_end is not None:
            c.dqn_eps_end = float(dqn_eps_end)
    if name == "ppo":
        c.ppo_eval = False
        c.rl_eval = False
        c.ppo_rollout_len = min(16, max(8, slots // 4))
    if name == "mappo":
        c.mappo_eval = False
        c.rl_eval = False
        c.mappo_rollout_len = min(16, max(8, slots // 4))
    return c


def model_names(config_or_agent):
    return [m["name"] for m in config_or_agent.models]


def action_pick_probs(agent):
    """Model-level pick probabilities (mean over users, sum over cell ranks)."""
    freq = np.asarray(agent.action_freq, dtype=float)
    mean_m = freq.sum(axis=2).mean(axis=0)
    s = float(mean_m.sum())
    if s > 0:
        mean_m = mean_m / s
    names = model_names(agent)
    return {names[i]: float(mean_m[i]) for i in range(len(names))}, freq


def run_one_sweep(name, cls, seed, slots, users, **knobs):
    c = configure_sweep(name, seed, slots, users, **knobs)
    t0 = time.time()
    agent = cls(c)
    agent.simulation()
    wall = time.time() - t0
    avg = agent.average_metrics
    pick, freq = action_pick_probs(agent)
    return {
        "name": name, "seed": seed,
        "reward": mean_task_reward(agent),
        "acc": float(avg["accuracy"]),
        "delay": float(avg["latency"]),
        "energy": float(avg["energy"]),
        "backlog": float(avg.get("backlog_bits", float("nan"))),
        "vio": float(avg["vio_prob"]),
        "vio_excess": float(avg.get("vio_excess", 0.0) or 0.0),
        "sec": wall,
        "pick": pick,
        "model_names": model_names(agent),
        "action_freq": freq,
        **algo_ms_per_slot(agent, slots, name=name),
    }


def sweep_outdir(root, sweep_name):
    d = os.path.join(root, sweep_name)
    os.makedirs(d, exist_ok=True)
    return d


def write_perseed_csv(path, rows, axis_key):
    fields = ["name", "seed", axis_key, "reward", "acc", "delay", "energy",
              "backlog", "vio", "vio_excess", "sec"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def aggregate_by_axis(rows, axis_key, algos):
    out = {}
    for algo in algos:
        xs = sorted({float(r[axis_key]) for r in rows if r["name"] == algo})
        out[algo] = {}
        for m in METRICS:
            means, stds = [], []
            for x in xs:
                vals = [float(r[m]) for r in rows
                        if r["name"] == algo and float(r[axis_key]) == x]
                means.append(float(np.mean(vals)) if vals else np.nan)
                stds.append(float(np.std(vals)) if vals else np.nan)
            out[algo][m] = (
                np.asarray(xs, dtype=float),
                np.asarray(means, dtype=float),
                np.asarray(stds, dtype=float),
            )
    return out


def plot_metric_vs_axis(agg, algos, axis_key, xlabel, out_dir, sweep_name):
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for algo in algos:
            if algo not in agg:
                continue
            x, mu, _sd = agg[algo][m]
            ax.plot(x, mu, marker="o", lw=1.8,
                    color=COLORS.get(algo), label=LABELS.get(algo, algo))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_YLABEL[m])
        ax.set_title(f"{sweep_name}: {m} vs {axis_key}")
        if m in LOG_Y_METRICS:
            pos = []
            for algo in algos:
                if algo not in agg:
                    continue
                _, mu, _ = agg[algo][m]
                pos.extend([float(v) for v in mu if np.isfinite(v) and v > 0])
            if pos:
                ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{m}_vs_{axis_key}.png"), dpi=160)
        plt.close(fig)


def load_sweep_perseed_csv(path, axis_key=None):
    if not os.path.isfile(path):
        return [], None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows_raw = list(reader)
        fields = reader.fieldnames or []
    if not rows_raw:
        return [], axis_key
    skip = {"name", "seed", "reward", "acc", "delay", "energy", "backlog",
            "vio", "vio_excess", "sec"}
    if axis_key is None:
        axis_key = next((c for c in fields if c not in skip), None)
    if axis_key is None:
        raise SystemExit(f"cannot infer sweep axis from {path}")
    rows = []
    for r in rows_raw:
        rows.append({
            "name": r["name"],
            "seed": int(r["seed"]),
            axis_key: float(r[axis_key]),
            "reward": float(r["reward"]),
            "acc": float(r["acc"]),
            "delay": float(r["delay"]),
            "energy": float(r["energy"]),
            "backlog": float(r["backlog"]),
            "vio": float(r["vio"]),
            "vio_excess": float(r.get("vio_excess", 0.0) or 0.0),
            "sec": float(r.get("sec", 0.0) or 0.0),
        })
    return rows, axis_key


def replot_sweep_dir(out_dir, sweep_name=None, axis_key=None, xlabel=None,
                     algos=None, csv_name="perseed.csv", mat_stem=None):
    out_dir = os.path.abspath(out_dir)
    sweep_name = sweep_name or os.path.basename(out_dir.rstrip(os.sep))
    csv_path = os.path.join(out_dir, csv_name)
    rows, axis_key = load_sweep_perseed_csv(csv_path, axis_key=axis_key)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    present = []
    for r in rows:
        if r["name"] not in present:
            present.append(r["name"])
    if algos:
        algos = [a for a in resolve_algos(algos) if a in present]
    else:
        algos = [a for a in ALGO_NAMES if a in present] + [
            a for a in present if a not in ALGO_NAMES]
    xlabel = xlabel or {
        "snr_db": "SNR [dB] (mean best-cell SINR)",
        "n_users": "# Mobile devices",
        "arrival_rate": "Arrival rate [tasks/slot]",
    }.get(axis_key, axis_key)
    plot_stem = mat_stem or sweep_name
    agg = aggregate_by_axis(rows, axis_key, algos)
    plot_metric_vs_axis(agg, algos, axis_key, xlabel, out_dir, plot_stem)
    mat_path = os.path.join(out_dir, f"{plot_stem}.mat")
    extra = load_mat_extras(mat_path)
    save_sweep_mat(mat_path, agg, algos, axis_key, rows, extra=extra or None)
    print(f"[replot] {plot_stem}: PNGs + {mat_path}", flush=True)
    return mat_path


def replot_action_pick_dir(out_dir, algos=None):
    """Redraw pick histograms from saved pick mats (no metric line plots)."""
    out_dir = os.path.abspath(out_dir)
    written = []
    for stem, title in (
        ("action_pick_snr", "Action pick probability vs SNR"),
        ("action_pick_users", "Action pick probability vs #MDs"),
    ):
        npz = os.path.join(out_dir, f"{stem}.npz")
        if not os.path.isfile(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        settings = list(z["settings"])
        models = list(z["models"])
        algo_list = list(z["algos"])
        if algos:
            algo_list = [a for a in resolve_algos(algos) if a in algo_list]
        picks = []
        for setting_i, setting in enumerate(settings):
            for algo in algo_list:
                mu = {}
                for m in models:
                    key = f"pick_{algo}_{m}"
                    if key in z.files:
                        mu[m] = float(z[key][setting_i])
                picks.append((str(setting), algo, mu))
        out_png = os.path.join(out_dir, f"{stem.replace('action_pick', 'action_pick_vs')}.png")
        # snr -> action_pick_vs_snr.png; users -> action_pick_vs_users.png
        if "snr" in stem:
            out_png = os.path.join(out_dir, "action_pick_vs_snr.png")
        else:
            out_png = os.path.join(out_dir, "action_pick_vs_users.png")
        plot_action_pick_bars(picks, models, out_png, title)
        written.append(out_png)
    if not written:
        raise SystemExit(
            f"no action_pick_*.npz under {out_dir}; re-run sweep_action_pick")
    return written


def save_sweep_mat(path, agg, algos, axis_key, rows, extra=None):
    payload = {
        "axis_key": axis_key,
        "algos": np.array(algos, dtype=object),
        "metrics": np.array(list(METRICS), dtype=object),
    }
    all_axes = sorted({float(r[axis_key]) for r in rows})
    payload["axis"] = np.asarray(all_axes, dtype=float)
    for algo in algos:
        if algo not in agg:
            continue
        for m in METRICS:
            x, mu, sd = agg[algo][m]
            mu_a = np.full(len(all_axes), np.nan)
            sd_a = np.full(len(all_axes), np.nan)
            for i, axv in enumerate(all_axes):
                idxs = np.where(np.isclose(x, axv))[0]
                if len(idxs):
                    mu_a[i] = mu[idxs[0]]
                    sd_a[i] = sd[idxs[0]]
            payload[f"{algo}_{m}_mean"] = mu_a
            payload[f"{algo}_{m}_std"] = sd_a
    payload["raw_name"] = np.array([r["name"] for r in rows], dtype=object)
    payload["raw_seed"] = np.asarray([r["seed"] for r in rows], dtype=np.int64)
    payload["raw_axis"] = np.asarray(
        [float(r[axis_key]) for r in rows], dtype=float)
    for m in METRICS:
        payload[f"raw_{m}"] = np.asarray([r[m] for r in rows], dtype=float)
    if extra:
        payload.update(extra)
    save_mat_and_python(path, payload, long_field_names=True)
    return path


def plot_action_pick_bars(pick_by_setting, model_order, out_path, title):
    """Grouped bars: one panel per algo, settings as bar groups over models."""
    algos = []
    for _, a, _ in pick_by_setting:
        if a not in algos:
            algos.append(a)
    n_algo = len(algos)
    fig, axes = plt.subplots(1, n_algo, figsize=(3.2 * n_algo, 4.0), sharey=True)
    if n_algo == 1:
        axes = [axes]
    settings = []
    for s, _, _ in pick_by_setting:
        if s not in settings:
            settings.append(s)
    x = np.arange(len(model_order))
    width = 0.8 / max(len(settings), 1)
    for ax, algo in zip(axes, algos):
        for si, setting in enumerate(settings):
            probs = None
            for s, a, p in pick_by_setting:
                if s == setting and a == algo:
                    probs = p
                    break
            if probs is None:
                continue
            vals = [probs.get(m, 0.0) for m in model_order]
            ax.bar(x + si * width, vals, width=width, label=str(setting))
        ax.set_xticks(x + width * (len(settings) - 1) / 2)
        ax.set_xticklabels(
            [MODEL_LABELS.get(m, m) for m in model_order],
            rotation=45, ha="right", fontsize=7)
        ax.set_title(LABELS.get(algo, algo), fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Action Pick Probability")
    axes[-1].legend(fontsize=7, title="setting")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_axis_sweep(
    sweep_name, axis_key, axis_values, *,
    knobs_for_value, xlabel, algos=None, seeds=(0, 1, 2),
    slots=DEFAULT_SLOTS, users=DEFAULT_USERS,
    out_root="figures/python figures/sweeps", collect_pick=False,
):
    algos = resolve_algos(list(algos) if algos else list(DEFAULT_SWEEP_ALGOS))
    cls_map = dict(ALGOS)
    out_dir = sweep_outdir(out_root, sweep_name)
    rows = []
    pick_records = []

    total = len(algos) * len(seeds) * len(axis_values)
    done = 0
    t_all = time.time()
    for v in axis_values:
        knobs = dict(knobs_for_value(v))
        users_v = int(knobs.pop("_users", users))
        slots_v = int(knobs.pop("_slots", slots))
        for name in algos:
            cls = cls_map[name]
            for seed in seeds:
                done += 1
                print(f"[{sweep_name} {done}/{total}] {name} seed={seed} "
                      f"{axis_key}={v} users={users_v} slots={slots_v}",
                      flush=True)
                r = run_one_sweep(name, cls, seed, slots_v, users_v, **knobs)
                r[axis_key] = float(v)
                print(f"    reward={r['reward']:.4f} delay={r['delay']:.3f} "
                      f"energy={r['energy']:.3f} acc={r['acc']:.4f} "
                      f"vio={r['vio']:.3f} backlog={r['backlog']:.0f} "
                      f"wall={r['sec']:.0f}s", flush=True)
                rows.append(r)
                if collect_pick:
                    pick_records.append((
                        float(v), name, int(seed), r["pick"], r["model_names"]))

    write_perseed_csv(os.path.join(out_dir, "perseed.csv"), rows, axis_key)
    agg = aggregate_by_axis(rows, axis_key, algos)
    plot_metric_vs_axis(agg, algos, axis_key, xlabel, out_dir, sweep_name)

    extra = {}
    if collect_pick and pick_records:
        model_order = pick_records[0][4]
        grouped = defaultdict(list)
        for axv, name, _seed, pick, _ in pick_records:
            grouped[(axv, name)].append(pick)
        axes_sorted = sorted({ax for ax, _ in grouped})
        for name in algos:
            for mname in model_order:
                arr = []
                for axv in axes_sorted:
                    plist = grouped.get((axv, name), [])
                    if not plist:
                        arr.append(np.nan)
                    else:
                        arr.append(float(np.mean([p[mname] for p in plist])))
                extra[f"pick_{name}_{mname}"] = np.asarray(arr, dtype=float)
        extra["pick_models"] = np.array(model_order, dtype=object)
        extra["pick_axis"] = np.asarray(axes_sorted, dtype=float)

    mat_path = os.path.join(out_dir, f"{sweep_name}.mat")
    save_sweep_mat(mat_path, agg, algos, axis_key, rows, extra=extra)
    elapsed = time.time() - t_all
    print(f"[{sweep_name}] done in {elapsed/60:.1f} min → {out_dir}/", flush=True)
    return rows, out_dir


def sweep_snr(algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS, users=DEFAULT_USERS,
              snr_targets=None, out_root="figures/python figures/sweeps"):
    snr_targets = list(snr_targets or [0, 2, 4, 6, 8, 10])

    def knobs(v):
        return {"sinr_offset_db": offset_for_snr_target(v, users=users)}

    return run_axis_sweep(
        "snr", "snr_db", snr_targets,
        knobs_for_value=knobs,
        xlabel="SNR [dB] (mean best-cell SINR)",
        algos=algos, seeds=seeds, slots=slots, users=users,
        out_root=out_root, collect_pick=True,
    )


def sweep_users(algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS, user_list=None,
                snr_db=5.0, out_root="figures/python figures/sweeps"):
    user_list = list(user_list or list(DEFAULT_USER_LIST))

    def knobs(v):
        return {
            "sinr_offset_db": offset_for_snr_target(snr_db, users=int(v)),
            "_users": int(v),
        }

    return run_axis_sweep(
        "users", "n_users", user_list,
        knobs_for_value=knobs,
        xlabel="Number of MDs",
        algos=algos, seeds=seeds, slots=slots, users=DEFAULT_USERS,
        out_root=out_root, collect_pick=True,
    )


def sweep_arrival(algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS,
                  users=DEFAULT_USERS, rates=None, snr_db=5.0,
                  out_root="figures/python figures/sweeps"):
    rates = list(rates or list(DEFAULT_ARRIVAL_RATES))
    off = offset_for_snr_target(snr_db, users=users)

    def knobs(v):
        return {"sinr_offset_db": off, "arrival_rate": float(v)}

    return run_axis_sweep(
        "arrival", "arrival_rate", rates,
        knobs_for_value=knobs,
        xlabel="Task arrival rate [tasks/slot]",
        algos=algos, seeds=seeds, slots=slots, users=users,
        out_root=out_root, collect_pick=False,
    )


def sweep_action_pick(
    algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS,
    snr_targets=None, user_list=None,
    out_root="figures/python figures/sweeps",
):
    """Pick histograms vs SNR and vs #MDs (paper Fig. 8)."""
    snr_targets = list(snr_targets or [2, 4, 6])
    user_list = list(user_list or [10, 15, 20])
    algos = resolve_algos(list(algos) if algos else list(DEFAULT_SWEEP_ALGOS))
    cls_map = dict(ALGOS)
    out_dir = sweep_outdir(out_root, "action_pick")

    rows_snr, rows_users = [], []
    pick_snr, pick_users = [], []
    pick_users_n = DEFAULT_USERS

    for snr in snr_targets:
        off = offset_for_snr_target(snr, users=pick_users_n)
        for name in algos:
            for seed in seeds:
                print(f"[action_pick/snr] {name} seed={seed} snr={snr}",
                      flush=True)
                r = run_one_sweep(
                    name, cls_map[name], seed, slots, pick_users_n,
                    sinr_offset_db=off)
                r["snr_db"] = float(snr)
                rows_snr.append(r)
                pick_snr.append(
                    (f"SNR={snr}", name, r["pick"], r["model_names"]))
                print(f"    reward={r['reward']:.3f} vio={r['vio']:.3f} "
                      f"pick_top={max(r['pick'], key=r['pick'].get)}",
                      flush=True)

    for nu in user_list:
        off_n = offset_for_snr_target(5.0, users=int(nu))
        for name in algos:
            for seed in seeds:
                print(f"[action_pick/users] {name} seed={seed} users={nu}",
                      flush=True)
                r = run_one_sweep(
                    name, cls_map[name], seed, slots, int(nu),
                    sinr_offset_db=off_n)
                r["n_users"] = float(nu)
                rows_users.append(r)
                pick_users.append(
                    (f"MDs={nu}", name, r["pick"], r["model_names"]))

    def _mean_picks(records):
        model_order = records[0][3]
        grouped = defaultdict(list)
        for setting, name, pick, _ in records:
            grouped[(setting, name)].append(pick)
        out = []
        for (setting, name), plist in grouped.items():
            mu = {k: float(np.mean([p[k] for p in plist])) for k in model_order}
            out.append((setting, name, mu))
        return out, model_order

    if pick_snr:
        mean_s, models = _mean_picks(pick_snr)
        plot_action_pick_bars(
            mean_s, models,
            os.path.join(out_dir, "action_pick_vs_snr.png"),
            "Action pick probability vs SNR",
        )
    if pick_users:
        mean_u, models = _mean_picks(pick_users)
        plot_action_pick_bars(
            mean_u, models,
            os.path.join(out_dir, "action_pick_vs_users.png"),
            "Action pick probability vs #MDs",
        )

    def _pick_mat(path, records, axis_name):
        if not records:
            return
        mean_recs, models = _mean_picks(records)
        settings = []
        for s, _, _ in mean_recs:
            if s not in settings:
                settings.append(s)
        payload = {
            "settings": np.array(settings, dtype=object),
            "models": np.array(models, dtype=object),
            "algos": np.array(algos, dtype=object),
            "axis_name": axis_name,
        }
        for algo in algos:
            for m in models:
                arr = []
                for s in settings:
                    found = None
                    for ss, aa, mu in mean_recs:
                        if ss == s and aa == algo:
                            found = mu.get(m, 0.0)
                            break
                    arr.append(np.nan if found is None else found)
                payload[f"pick_{algo}_{m}"] = np.asarray(arr, dtype=float)
        save_mat_and_python(path, payload, long_field_names=True)

    _pick_mat(os.path.join(out_dir, "action_pick_snr.mat"), pick_snr, "snr")
    _pick_mat(os.path.join(out_dir, "action_pick_users.mat"), pick_users, "users")

    if rows_snr:
        write_perseed_csv(
            os.path.join(out_dir, "perseed_snr.csv"), rows_snr, "snr_db")
    if rows_users:
        write_perseed_csv(
            os.path.join(out_dir, "perseed_users.csv"), rows_users, "n_users")

    print(f"[action_pick] done → {out_dir}/", flush=True)
    return {"snr": rows_snr, "users": rows_users}, out_dir


def add_common_args(p, *, default_algos=None):
    choices = list(ALGO_NAMES) + ["all"]
    p.add_argument(
        "--algos", nargs="+",
        default=list(default_algos or DEFAULT_SWEEP_ALGOS),
        metavar="NAME",
        help=f"schemes to run; 'all' or omit for full set. Choices: "
             f"{', '.join(choices)}",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    p.add_argument("--users", type=int, default=DEFAULT_USERS)
    p.add_argument("--out-root", default="figures/python figures/sweeps")
    return p


def resolved_algos_from_args(args):
    return resolve_algos(getattr(args, "algos", None))
