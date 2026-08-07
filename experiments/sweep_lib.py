"""Shared parameter-sweep evaluation for OMNIS schemes.

Honest eval: **re-runs online simulation** for every (algo, seed, setting).
MAB / most DRL checkpoints are not persisted by ``train_all`` (only DQN has
optional ``save_pretrained`` after an explicit ``pretrain()`` path, which
``train_all`` does not call). Do not invent fake checkpoints.

Metrics (aligned with paper Sec. VI + journal queueing extension):
  reward   — mean Lyapunov V·u + drift (same as ``train_lib.reward_series``)
  delay    — mean latency [s]
  energy   — mean energy [J]
  acc      — mean inference accuracy
  vio      — QoS violation probability (delay|energy); paper ``Avg. Violation Prob.``
             (not a separate accuracy-violation metric)
  backlog  — mean backlog [bits] (journal extension; not in conference paper)

Outputs under ``figures/sweeps/<sweep_name>/``: PNGs + ``.mat`` + CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict

# Headless plotting (overnight / SSH / Cursor agents)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from experiments.train_lib import (
    ALGOS,
    ALGO_NAMES,
    algo_ms_per_slot,
    mean_series,
    resolve_algos,
    reward_series,
    cumulative_mean,
)
from sys_data.config import Config

# Paper Fig. 6 SNR axis: offset maps target ≈ mean *best-cell* / serving SINR.
# Decisions use top-L / best-cell SINR (~+21 dB above the all-cell mean), so
# calibrating to the all-cell mean left Acc flat near the MCS ceiling.
# Calibration uses the fixed max UE pool (25) so users/snr sweeps share one
# offset; nested n∈{5,10,…,25} are prefixes of that pool.
TRACE_MEAN_SINR_DB = -10.75           # smoke7_sites all-cell empirical mean (legacy)
SWEEP_UE_POOL_SIZE = 25
DEFAULT_SLOTS = 400
DEFAULT_USERS = 10                    # SNR / arrival default MD count
DEFAULT_USER_LIST = (5, 10, 15, 20, 25)
DEFAULT_SWEEP_ALGOS = ("causal", "ucb", "gdo", "dqn", "ppo")
METRICS = ("reward", "delay", "energy", "acc", "vio", "backlog")
# Log-y helps when Acc-chasing / heavy-queue algos crush linear scale.
LOG_Y_METRICS = frozenset({"delay", "backlog"})
METRIC_YLABEL = {
    "reward": "Avg. Reward (V·u+drift)",
    "delay": "Avg. Latency [s]",
    "energy": "Avg. Energy [J]",
    "acc": "Avg. Acc.",
    "vio": "Avg. Violation Prob. (delay|energy)",
    "backlog": "Avg. Backlog [bits]",
}
LABELS = {
    "causal": "OMNIS-Causal", "ucb": "OMNIS-UCB",
    "dqn": "DQN (joint)", "ppo": "PPO", "mappo": "MAPPO",
    "gdo": "GDO", "rss": "RSS", "dts": "OMNIS-TS", "cto": "CTO",
}
COLORS = {
    "causal": "#1f77b4", "ucb": "#ff7f0e", "dqn": "#2ca02c",
    "ppo": "#bcbd22", "mappo": "#e377c2",
    "gdo": "#d62728", "rss": "#9467bd", "dts": "#8c564b", "cto": "#17becf",
}
MODEL_LABELS = {
    "Box3": "Box-3", "Box6": "Box-6", "Box12": "Box-12",
    "Standard3": "Standard-3", "Standard6": "Standard-6", "Standard12": "Standard-12",
}


def _mean_best_cell_sinr_db(ue_ids, tag="smoke7", table_dir="phy_sim/output"):
    """Empirical mean of max-cell SINR [dB] over slots×UEs for a fixed pool."""
    from omnis.sinr_trace import SinrTrace
    tr = SinrTrace.from_config_dir(table_dir, tag=tag, ue_ids=list(ue_ids))
    # sinr_db: [slots, cells, ues] → max over cells, mean over slots×ues
    return float(np.mean(np.max(tr.sinr_db, axis=1)))


def fixed_ue_pool(pool_size=SWEEP_UE_POOL_SIZE, tag="smoke7",
                  table_dir="phy_sim/output"):
    """Stable max UE pool (same as Config.sinr_ue_pool)."""
    from omnis.sinr_trace import SinrTrace, select_spread_ue_ids
    probe = SinrTrace.from_config_dir(table_dir, tag=tag)
    return select_spread_ue_ids(int(pool_size), probe.num_ues, probe.num_cells)


# Calibrate once on the fixed 25-UE pool (nested users share this offset).
_UE_POOL_25 = fixed_ue_pool(SWEEP_UE_POOL_SIZE)
TRACE_MEAN_BEST_CELL_SINR_DB = _mean_best_cell_sinr_db(_UE_POOL_25)


def offset_for_snr_target(snr_target_db: float) -> float:
    """Additive dB shift so mean *best-cell* trace SINR ≈ ``snr_target_db``.

    Paper Fig. 6 axis is labeled SNR [dB]; the simulator associates each MD to
    its strongest (or top-L) cell. Offset is calibrated on the fixed 25-UE
    pool so users / snr sweeps stay consistent under nested prefixes.
    """
    return float(snr_target_db) - TRACE_MEAN_BEST_CELL_SINR_DB


def configure_sweep(
    name,
    seed,
    slots,
    users,
    *,
    sinr_offset_db=0.0,
    arrival_rate=None,
    arrival_scale=None,
    causal_beta=None,
    beta_const=None,
    dqn_eps_end=None,
):
    """Build Config like ``train_lib.configure``, plus sweep knobs."""
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.sinr_offset_db = float(sinr_offset_db)

    if arrival_rate is not None:
        rate = float(arrival_rate)
        c.arrival_rate = {u: rate for u in c.users}
    elif arrival_scale is not None:
        scale = float(arrival_scale)
        c.arrival_rate = {
            u: float(c.arrival_rate_origin[u]) * scale for u in c.users
        }

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
    """Return model-level pick probabilities (mean over users, sum cell ranks).

    ``agent.action_freq`` is [U, n_models, L] already normalized by slots
    (fraction of slots each (model, cell_rank) was chosen per user).
    """
    freq = np.asarray(agent.action_freq, dtype=float)
    # Per-user model probs: sum over cell ranks → [U, n_models]
    per_user = freq.sum(axis=2)
    # Mean across users → [n_models]; already a probability (sums to ~1)
    mean_m = per_user.mean(axis=0)
    s = float(mean_m.sum())
    if s > 0:
        mean_m = mean_m / s
    names = model_names(agent)
    return {names[i]: float(mean_m[i]) for i in range(len(names))}, freq


def run_one_sweep(name, cls, seed, slots, users, **knobs):
    """Run one online simulation; return scalars + action pick probs."""
    c = configure_sweep(name, seed, slots, users, **knobs)
    t0 = time.time()
    agent = cls(c)
    agent.comm_rtt_s = getattr(c, "comm_rtt_s", 1e-3)
    agent.comm_ctrl_rate_bps = getattr(c, "comm_ctrl_rate_bps", 1e6)
    agent.simulation()
    wall = time.time() - t0
    avg = agent.average_metrics
    obj = reward_series(agent)
    timing = algo_ms_per_slot(agent, slots, name=name)
    pick, freq = action_pick_probs(agent)
    return {
        "name": name,
        "seed": int(seed),
        "reward": float(np.mean(obj)),
        "acc": float(avg["accuracy"]),
        "delay": float(avg["latency"]),
        "energy": float(avg["energy"]),
        "backlog": float(avg.get("backlog_bits", float("nan"))),
        "vio": float(avg["vio_prob"]),
        "vio_excess": float(avg.get("vio_sum", float("nan"))),
        "pick": pick,
        "action_freq": freq,
        "model_names": model_names(agent),
        "sec": wall,
        "rew_series": obj,
        "cum_reward": cumulative_mean(obj),
        "acc_series": mean_series(agent, "accuracy"),
        "delay_series": mean_series(agent, "delay"),
        "energy_series": mean_series(agent, "energy"),
        "backlog_series": mean_series(agent, "backlog"),
        "vio_series": mean_series(agent, "is_vio"),
        **timing,
        **{k: knobs.get(k) for k in (
            "sinr_offset_db", "arrival_rate", "arrival_scale",
            "causal_beta", "beta_const", "dqn_eps_end",
        )},
    }


def sweep_outdir(root, sweep_name):
    d = os.path.join(root, sweep_name)
    os.makedirs(d, exist_ok=True)
    return d


def write_perseed_csv(path, rows, axis_key, axis_values_in_row=True):
    fields = [
        "name", "seed", axis_key,
        "reward", "acc", "delay", "energy", "backlog", "vio", "vio_excess", "sec",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fields}
            for k in ("reward", "acc", "delay", "energy", "backlog", "vio",
                      "vio_excess", "sec", axis_key):
                if out.get(k) is not None and isinstance(out[k], float):
                    out[k] = f"{out[k]:.6g}"
            w.writerow(out)


def aggregate_by_axis(rows, axis_key, algos):
    """Return dict[algo][metric] -> (axis_sorted, mean, std) over seeds."""
    buckets = defaultdict(lambda: defaultdict(list))  # (algo, axis) -> metrics
    for r in rows:
        ax = float(r[axis_key])
        buckets[(r["name"], ax)]["reward"].append(r["reward"])
        buckets[(r["name"], ax)]["delay"].append(r["delay"])
        buckets[(r["name"], ax)]["energy"].append(r["energy"])
        buckets[(r["name"], ax)]["acc"].append(r["acc"])
        buckets[(r["name"], ax)]["vio"].append(r["vio"])
        buckets[(r["name"], ax)]["backlog"].append(r["backlog"])

    out = {}
    for algo in algos:
        axes = sorted({ax for (n, ax) in buckets if n == algo})
        out[algo] = {}
        for m in METRICS:
            means, stds = [], []
            for ax in axes:
                v = np.asarray(buckets[(algo, ax)][m], dtype=float)
                means.append(float(v.mean()))
                stds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            out[algo][m] = (
                np.asarray(axes, dtype=float),
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
            x, mu, sd = agg[algo][m]
            color = COLORS.get(algo, None)
            label = LABELS.get(algo, algo)
            # Mean line + markers only (std still stored in .mat / aggregate).
            ax.plot(x, mu, marker="o", lw=1.8, color=color, label=label)
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


def save_sweep_mat(path, agg, algos, axis_key, rows, extra=None):
    """MATLAB-friendly mat: axis + per-algo mean/std arrays + raw rows."""
    payload = {
        "axis_key": axis_key,
        "algos": np.array(algos, dtype=object),
        "metrics": np.array(list(METRICS), dtype=object),
    }
    # Common axis (union)
    all_axes = sorted({float(r[axis_key]) for r in rows})
    payload["axis"] = np.asarray(all_axes, dtype=float)

    for algo in algos:
        if algo not in agg:
            continue
        for m in METRICS:
            x, mu, sd = agg[algo][m]
            # Align to common axis
            mu_a = np.full(len(all_axes), np.nan)
            sd_a = np.full(len(all_axes), np.nan)
            for i, axv in enumerate(all_axes):
                idxs = np.where(np.isclose(x, axv))[0]
                if len(idxs):
                    mu_a[i] = mu[idxs[0]]
                    sd_a[i] = sd[idxs[0]]
            payload[f"{algo}_{m}_mean"] = mu_a
            payload[f"{algo}_{m}_std"] = sd_a

    # Raw per-seed table
    payload["raw_name"] = np.array([r["name"] for r in rows], dtype=object)
    payload["raw_seed"] = np.asarray([r["seed"] for r in rows], dtype=np.int64)
    payload["raw_axis"] = np.asarray([float(r[axis_key]) for r in rows], dtype=float)
    for m in METRICS:
        payload[f"raw_{m}"] = np.asarray([r[m] for r in rows], dtype=float)

    if extra:
        payload.update(extra)
    savemat(path, payload, long_field_names=True, do_compression=True)
    return path


def plot_action_pick_bars(pick_by_setting, model_order, out_path, title):
    """Grouped bar: settings × models, one panel per algo or combined.

    ``pick_by_setting``: list of (setting_label, algo, dict model->prob)
    We make one figure per algo with bars over models for each setting.
    """
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
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_order],
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
    sweep_name,
    axis_key,
    axis_values,
    *,
    knobs_for_value,
    xlabel,
    algos=None,
    seeds=(0, 1, 2),
    slots=DEFAULT_SLOTS,
    users=DEFAULT_USERS,
    out_root="figures/sweeps",
    collect_pick=False,
):
    """Generic sweep over one scalar axis.

    ``knobs_for_value(v)`` returns kwargs for ``run_one_sweep`` (sinr_offset_db, …)
    and may also set users if needed via returning ``_users`` override.
    """
    algos = resolve_algos(list(algos) if algos else list(DEFAULT_SWEEP_ALGOS))
    cls_map = dict(ALGOS)
    out_dir = sweep_outdir(out_root, sweep_name)
    rows = []
    pick_records = []  # (axis, algo, seed, pick_dict, model_names)

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

    # Aggregate / plot / mat
    write_perseed_csv(os.path.join(out_dir, "perseed.csv"), rows, axis_key)
    agg = aggregate_by_axis(rows, axis_key, algos)
    plot_metric_vs_axis(agg, algos, axis_key, xlabel, out_dir, sweep_name)

    extra = {}
    if collect_pick and pick_records:
        # Mean pick over seeds: (axis, algo) -> model probs
        model_order = pick_records[0][4]
        grouped = defaultdict(list)
        for axv, name, seed, pick, _ in pick_records:
            grouped[(axv, name)].append(pick)
        mean_picks = []
        pick_mat = {}
        for (axv, name), plist in grouped.items():
            keys = model_order
            mu = {k: float(np.mean([p[k] for p in plist])) for k in keys}
            mean_picks.append((f"{axis_key}={axv:g}", name, mu))
            for k in keys:
                pick_mat.setdefault(f"pick_{name}_{k}", [])
            # filled below on common axis
        # Build pick arrays aligned to axis
        axes_sorted = sorted({ax for ax, _ in grouped})
        for name in algos:
            for mi, mname in enumerate(model_order):
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

        # Bar plot at each axis value (paper Fig. 8 style)
        for axv in axes_sorted:
            subset = [(f"{axis_key}={axv:g}", a, mu)
                      for (s, a, mu) in mean_picks if s == f"{axis_key}={axv:g}"]
            if subset:
                plot_action_pick_bars(
                    subset, model_order,
                    os.path.join(out_dir, f"action_pick_{axis_key}{axv:g}.png"),
                    f"Action pick @ {axis_key}={axv:g}",
                )
        # Combined: all settings, per algo
        plot_action_pick_bars(
            mean_picks, model_order,
            os.path.join(out_dir, "action_pick_all.png"),
            f"{sweep_name}: action pick probability",
        )

    mat_path = os.path.join(out_dir, f"{sweep_name}.mat")
    save_sweep_mat(mat_path, agg, algos, axis_key, rows, extra=extra)
    elapsed = time.time() - t_all
    print(f"[{sweep_name}] done in {elapsed/60:.1f} min → {out_dir}/", flush=True)
    return rows, out_dir


# ---- concrete sweeps (paper Sec. VI + journal arrival) ----

def sweep_snr(algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS, users=DEFAULT_USERS,
              snr_targets=None, out_root="figures/sweeps"):
    """Paper Fig. 6: metrics vs SNR [dB] (via best-cell-calibrated sinr_offset_db)."""
    snr_targets = list(snr_targets or [0, 2, 4, 6, 8, 10])

    def knobs(v):
        return {"sinr_offset_db": offset_for_snr_target(v)}

    return run_axis_sweep(
        "snr", "snr_db", snr_targets,
        knobs_for_value=knobs,
        xlabel="SNR [dB] (mean best-cell SINR)",
        algos=algos, seeds=seeds, slots=slots, users=users,
        out_root=out_root, collect_pick=True,
    )


def sweep_users(algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS, user_list=None,
                snr_db=5.0, out_root="figures/sweeps"):
    """Paper Fig. 7: metrics vs number of MDs (nested prefixes of 25-UE pool)."""
    user_list = list(user_list or list(DEFAULT_USER_LIST))
    off = offset_for_snr_target(snr_db)

    def knobs(v):
        return {"sinr_offset_db": off, "_users": int(v)}

    return run_axis_sweep(
        "users", "n_users", user_list,
        knobs_for_value=knobs,
        xlabel="Number of MDs",
        algos=algos, seeds=seeds, slots=slots, users=DEFAULT_USERS,
        out_root=out_root, collect_pick=True,
    )


def sweep_arrival(algos=None, seeds=(0, 1, 2), slots=DEFAULT_SLOTS,
                  users=DEFAULT_USERS,
                  rates=None, snr_db=5.0, out_root="figures/sweeps"):
    """Journal extension: metrics vs uniform Poisson arrival rate [tasks/slot]."""
    rates = list(rates or [0.30, 0.50, 0.70, 0.90, 1.10])
    off = offset_for_snr_target(snr_db)

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
    beta_values=None,
    out_root="figures/sweeps",
):
    """Paper Fig. 8 style + exploration-knob sweep.

    (a) Export action pick at SNR∈{2,4,6} and MDs∈{10,15,20}.
    (b) Sweep Causal ``causal_beta`` / UCB ``beta_const`` / DQN ``eps_end``
        (sparser β grid to keep wall time manageable at 400 slots).
    """
    snr_targets = list(snr_targets or [2, 4, 6])
    user_list = list(user_list or [10, 15, 20])
    beta_values = list(beta_values or [0.55, 1.0, 2.0])
    algos = resolve_algos(list(algos) if algos else list(DEFAULT_SWEEP_ALGOS))
    cls_map = dict(ALGOS)
    out_dir = sweep_outdir(out_root, "action_pick")

    rows_snr = []
    rows_users = []
    rows_beta = []
    pick_snr = []
    pick_users = []

    # --- (a1) SNR grid @ DEFAULT_USERS (harder load so picks/metrics move) ---
    pick_users_n = DEFAULT_USERS
    for snr in snr_targets:
        off = offset_for_snr_target(snr)
        for name in algos:
            for seed in seeds:
                print(f"[action_pick/snr] {name} seed={seed} snr={snr}", flush=True)
                r = run_one_sweep(name, cls_map[name], seed, slots, pick_users_n,
                                  sinr_offset_db=off)
                r["snr_db"] = float(snr)
                rows_snr.append(r)
                pick_snr.append((f"SNR={snr}", name, r["pick"], r["model_names"]))
                print(f"    reward={r['reward']:.3f} vio={r['vio']:.3f} "
                      f"pick_top={max(r['pick'], key=r['pick'].get)}", flush=True)

    # --- (a2) user grid @ SNR=5 ---
    off5 = offset_for_snr_target(5.0)
    for nu in user_list:
        for name in algos:
            for seed in seeds:
                print(f"[action_pick/users] {name} seed={seed} users={nu}", flush=True)
                r = run_one_sweep(name, cls_map[name], seed, slots, int(nu),
                                  sinr_offset_db=off5)
                r["n_users"] = float(nu)
                rows_users.append(r)
                pick_users.append((f"MDs={nu}", name, r["pick"], r["model_names"]))

    # --- (b) exploration knob ---
    for beta in beta_values:
        for name in algos:
            for seed in seeds:
                kn = {"sinr_offset_db": off5}
                if name == "causal":
                    kn["causal_beta"] = float(beta)
                elif name in ("ucb", "dts", "cto"):
                    kn["beta_const"] = float(beta)
                elif name == "dqn":
                    kn["dqn_eps_end"] = float(min(max(beta / 4.0, 0.01), 0.5))
                print(f"[action_pick/beta] {name} seed={seed} beta={beta}", flush=True)
                r = run_one_sweep(name, cls_map[name], seed, slots, pick_users_n, **kn)
                r["explore_knob"] = float(beta)
                rows_beta.append(r)
                print(f"    reward={r['reward']:.3f} vio={r['vio']:.3f}", flush=True)

    # Aggregate mean picks over seeds for bar plots
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
            "Action pick probability vs SNR (paper Fig. 8)",
        )
    if pick_users:
        mean_u, models = _mean_picks(pick_users)
        plot_action_pick_bars(
            mean_u, models,
            os.path.join(out_dir, "action_pick_vs_users.png"),
            "Action pick probability vs #MDs (paper Fig. 8)",
        )

    # Metrics vs explore knob
    if rows_beta:
        write_perseed_csv(os.path.join(out_dir, "perseed_explore.csv"),
                          rows_beta, "explore_knob")
        agg = aggregate_by_axis(rows_beta, "explore_knob", algos)
        plot_metric_vs_axis(
            agg, algos, "explore_knob",
            "Exploration knob (causal_beta / UCB β / ~4·DQN ε_end)",
            out_dir, "action_pick_explore",
        )
        save_sweep_mat(
            os.path.join(out_dir, "action_pick_explore.mat"),
            agg, algos, "explore_knob", rows_beta,
        )

    # Save pick mats
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
        savemat(path, payload, long_field_names=True, do_compression=True)

    _pick_mat(os.path.join(out_dir, "action_pick_snr.mat"), pick_snr, "snr")
    _pick_mat(os.path.join(out_dir, "action_pick_users.mat"), pick_users, "users")

    # Also dump perseed for snr/users grids
    if rows_snr:
        write_perseed_csv(os.path.join(out_dir, "perseed_snr.csv"), rows_snr, "snr_db")
    if rows_users:
        write_perseed_csv(os.path.join(out_dir, "perseed_users.csv"),
                          rows_users, "n_users")

    print(f"[action_pick] done → {out_dir}/", flush=True)
    return {
        "snr": rows_snr, "users": rows_users, "explore": rows_beta,
    }, out_dir


def add_common_args(p):
    p.add_argument("--algos", nargs="+", default=list(DEFAULT_SWEEP_ALGOS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    p.add_argument("--users", type=int, default=DEFAULT_USERS)
    p.add_argument("--out-root", default="figures/sweeps")
    return p
