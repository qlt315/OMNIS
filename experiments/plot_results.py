"""Plot convergence figures from perseed.csv + series/*.npz (keeps source CSV)."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from repo_util import ensure_repo_root
except ImportError:
    from experiments.repo_util import ensure_repo_root  # type: ignore

ensure_repo_root()

from convergence_lib import ALGO_NAMES, SERIES_KEYS, series_dir  # noqa: E402
try:
    from comm_model import comm_ms_per_slot
except ImportError:  # when imported as experiments.plot_results
    from experiments.comm_model import comm_ms_per_slot  # noqa: E402
try:
    from result_io import save_mat_and_python
except ImportError:
    from experiments.result_io import save_mat_and_python  # noqa: E402

# =============================================================================
# PyCharm defaults (CLI flags override these)
# =============================================================================
PYCHARM_INDIR = "figures/python figures/convergence"
PYCHARM_OUT = None                 # None → same as indir
PYCHARM_ALGOS = None               # None → all in CSV; or ["causal","ucb"] / "all"
PYCHARM_SLIDE = 5
PYCHARM_USERS = None               # None → Config.user_num for comm_ms backfill
PYCHARM_WRITE_MAT = True
# =============================================================================

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

SCALAR_METRICS = (
    "reward", "acc", "delay", "energy", "backlog", "vio",
    "ms_per_slot", "decision_ms", "select_ms", "comm_ms", "bcd_ms",
    "bcd_ms_wall", "update_ms",
    "comm_uplink_B", "comm_downlink_B", "comm_rounds", "sec",
)

# Stacked runtime: selection + learning update + interaction + resource alloc.
# Use median across seeds (mean is dominated by rare wall-clock outliers).
RUNTIME_STACK = ("select_ms", "update_ms", "comm_ms", "bcd_ms")
RUNTIME_STACK_LABELS = (
    "selection", "update", "interaction", "resource allocation")
RUNTIME_STACK_COLORS = ("#4c78a8", "#9ecae9", "#54a24b", "#f58518")


def sliding_mean(x, w=5):
    """Trailing window mean; NaNs (idle slots) are skipped in the window."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < w:
        return x.copy()
    out = np.empty(n - w + 1, dtype=float)
    for i in range(out.size):
        wdw = x[i:i + w]
        m = np.isfinite(wdw)
        out[i] = float(np.mean(wdw[m])) if m.any() else np.nan
    return out


def running_mean(x):
    """Cumulative average; NaNs (idle slots) skipped in sum and count."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    valid = np.isfinite(x)
    csum = np.cumsum(np.where(valid, x, 0.0))
    count = np.cumsum(valid.astype(float))
    out = np.full_like(x, np.nan, dtype=float)
    m = count > 0
    out[m] = csum[m] / count[m]
    return out


def windowed_running_mean(x, w):
    """Trailing-window average of width ``w``; NaNs skipped."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    w = max(1, min(int(w), n))
    out = np.empty(n, dtype=float)
    for t in range(n):
        a = max(0, t - w + 1)
        wdw = x[a:t + 1]
        m = np.isfinite(wdw)
        out[t] = float(np.mean(wdw[m])) if m.any() else np.nan
    return out


def windowed_running_mean_from_zero(x, w):
    """Trailing-window mean that starts at 0.

    Idle NaNs → 0, and the window is left-padded with ``w-1`` zeros so the
    first points are near the origin and the average rises over ~``w`` slots
    instead of jumping to the first completion's score.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n = x.size
    if n == 0:
        return x.copy()
    w = max(1, int(w))
    pad = np.zeros(w - 1, dtype=float)
    xp = np.concatenate([pad, x])
    c = np.cumsum(xp)
    # window sum ending at index (w-1+t) in xp == index t in x
    out = np.empty(n, dtype=float)
    for t in range(n):
        end = w - 1 + t
        start = end - w
        out[t] = (c[end] - (c[start] if start >= 0 else 0.0)) / float(w)
    return out



def bandplot(ax, series_list, color, label, sliding=None, lw=1.8, alpha=0.15,
             robust=False, band=True):
    """Plot seed curves as a center line + optional band.

    ``robust=False``: mean ± std (legacy).
    ``robust=True``: per-seed p5–p95 winsorize, then median + IQR [p25, p75].
    Winsorize removes single-slot explosions; IQR resists a bad seed.
    ``band=False``: center line only (readable when many noisy schemes share axes).
    """
    if not series_list:
        return
    L = min(len(s) for s in series_list)
    arr = np.stack([np.asarray(s[:L], dtype=float) for s in series_list], axis=0)
    if sliding:
        arr = np.stack([sliding_mean(row, sliding) for row in arr], axis=0)
        x = np.arange(arr.shape[1]) + sliding - 1
    else:
        x = np.arange(arr.shape[1])
    if robust and arr.shape[0] > 1:
        for i in range(arr.shape[0]):
            lo_i, hi_i = np.percentile(arr[i], [5, 95])
            if np.isfinite(lo_i) and np.isfinite(hi_i) and hi_i > lo_i:
                arr[i] = np.clip(arr[i], lo_i, hi_i)
        m = np.median(arr, axis=0)
        lo = np.percentile(arr, 25, axis=0)
        hi = np.percentile(arr, 75, axis=0)
    else:
        m = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
        lo, hi = m - sd, m + sd
    ax.plot(x, m, color=color, lw=lw, label=label)
    if band and alpha > 0:
        ax.fill_between(x, lo, hi, color=color, alpha=alpha)


def _autoscale_ylim(ax, pad_frac=0.10, abs_pad=0.6, q_lo=10.0, q_hi=90.0,
                    skip_first=0, skip_frac=0.0):
    """Y-limits from curve percentiles; optional early-sample skip."""
    ys = []
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        if not y.size:
            continue
        y = y[np.isfinite(y)]
        n_skip = int(skip_first)
        if skip_frac > 0 and y.size:
            n_skip = max(n_skip, int(round(skip_frac * y.size)))
        if n_skip > 0 and y.size > n_skip + 1:
            y = y[n_skip:]
        if y.size:
            ys.append(y)
    if not ys:
        return
    y = np.concatenate(ys)
    if y.size < 2:
        return
    lo = float(np.percentile(y, q_lo))
    hi = float(np.percentile(y, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(y)), float(np.max(y))
    pad = max((hi - lo) * pad_frac, abs_pad)
    ax.set_ylim(lo - pad, hi + pad)


def _comm_defaults(user_num=None):
    """RTT / rate / user_num / local_obs_dim (match Config; no full PHY init)."""
    # Defaults aligned with sys_data.config.Config control-plane fields.
    rtt_s, rate, users, top_l, num_cells = 1e-3, 1e6, 10, 3, 7
    try:
        # Prefer live Config attrs when available without constructing Config()
        # (Config.__init__ loads SINR traces). Fall back to parsing source.
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sys_data", "config.py")
        with open(cfg_path, encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"self\.comm_rtt_s\s*=\s*([0-9.eE+-]+)", text)
        if m:
            rtt_s = float(m.group(1))
        m = re.search(r"self\.comm_ctrl_rate_bps\s*=\s*([0-9.eE+-]+)", text)
        if m:
            rate = float(m.group(1))
        # Do not parse Config.user_num / ue_pool_size here (pool=25 ≠ train U).
        # Prefer CLI --users; else default 10 (convergence_all default).
        m = re.search(r"self\.top_l_cells\s*=\s*(\d+)", text)
        if m:
            top_l = int(m.group(1))
        m = re.search(r"self\.num_cells\s*=\s*(\d+)", text)
        if m:
            num_cells = int(m.group(1))
    except Exception as e:
        print(f"warning: could not read config.py ({e}); using comm defaults",
              flush=True)
    if user_num is not None:
        users = int(user_num)
    # Same estimate as convergence_lib.algo_ms_per_slot when agent has no local_obs_dim
    local_obs_dim = 4 + top_l + 2
    return {
        "rtt_s": rtt_s,
        "ctrl_rate_bps": rate,
        "user_num": users,
        "local_obs_dim": local_obs_dim,
        "num_cells": num_cells,
    }


def _approx_eq(a, b, rtol=1e-3, atol=1e-3):
    return abs(float(a) - float(b)) <= atol + rtol * max(abs(float(a)), abs(float(b)), 1e-12)


def enrich_runtime_rows(rows, user_num=None):
    """Fill / normalize runtime fields for stacking (old + new CSV formats).

    New convergence_lib: decision_ms already includes update; ms = decision+comm+bcd.
    Old CSV: decision_ms is decision-only; ms = decision+bcd+update; no comm_*.

    Stack segments: selection + update + interaction/comm + per-algo
    resource-allocation wall (BCD or GDO EG slice; CSV field ``bcd_ms``).
    """
    cfg = _comm_defaults(user_num=user_num)

    by_algo = {}
    for r in rows:
        name = r["name"]
        v = r.get("bcd_ms")
        if v is None or v == "":
            continue
        by_algo.setdefault(name, []).append(float(v))
    if by_algo:
        algo_meds = {k: float(np.median(v)) for k, v in by_algo.items()}
        meds = list(algo_meds.values())
        spread = max(meds) - min(meds)
        if spread > 2.0:  # ms
            parts = ", ".join(f"{k}={v:.1f}" for k, v in sorted(algo_meds.items()))
            print(
                f"note: resource-allocation wall medians differ across algos "
                f"(spread={spread:.1f}ms: {parts}); "
                f"runtime keeps per-algo walls",
                flush=True,
            )

    for r in rows:
        name = r["name"]
        dec = float(r.get("decision_ms", 0.0) or 0.0)
        bcd_wall = float(r.get("bcd_ms", 0.0) or 0.0)
        upd = float(r.get("update_ms", 0.0) or 0.0)
        ms = float(r.get("ms_per_slot", 0.0) or 0.0)
        has_comm = "comm_ms" in r and r["comm_ms"] is not None

        # Always recompute interaction from the control-plane model so protocol
        # changes (rounds / payloads) apply without a full retrain. Decision
        # and resource allocation stay measured per algorithm.
        comm = comm_ms_per_slot(
            name,
            user_num=cfg["user_num"],
            local_obs_dim=cfg["local_obs_dim"],
            rtt_s=cfg["rtt_s"],
            ctrl_rate_bps=cfg["ctrl_rate_bps"],
            num_cells=cfg.get("num_cells", 7),
        )
        if has_comm:
            # New format: decision_ms already folds update into compute time.
            decision_stack = dec
        else:
            # Old: ms ≈ decision + bcd + update (decision excludes update).
            if _approx_eq(ms, dec + bcd_wall + upd):
                decision_stack = dec + upd
            else:
                decision_stack = dec

        r["decision_ms"] = float(decision_stack)
        r["bcd_ms_wall"] = bcd_wall
        r["bcd_ms"] = float(bcd_wall)
        r["update_ms"] = upd  # parallel learning update [ms/slot]
        # selection-only = folded decision minus update (floored at 0)
        r["select_ms"] = float(max(decision_stack - upd, 0.0))
        r["comm_ms"] = float(comm["comm_ms"])
        r["comm_uplink_B"] = float(comm["comm_uplink_B"])
        r["comm_downlink_B"] = float(comm["comm_downlink_B"])
        r["comm_rounds"] = float(comm["comm_rounds"])
        r["ms_per_slot"] = float(
            decision_stack + bcd_wall + float(comm["comm_ms"]))
    return rows, cfg


def load_perseed(indir):
    path = os.path.join(indir, "perseed.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}; run a train_* script first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        for k, v in list(r.items()):
            if k in ("name", "seed"):
                continue
            if v is None or v == "":
                del r[k]
                continue
            r[k] = float(v)
    return rows


def discover_algos(rows, series_root, want=None):
    """Algos present in CSV (and preferably series); honor optional filter."""
    present = []
    for n in ALGO_NAMES:
        if want is not None and n not in want:
            continue
        if not any(r["name"] == n for r in rows):
            continue
        present.append(n)
    # extras not in ALGO_NAMES
    for r in rows:
        n = r["name"]
        if n not in present and (want is None or n in want):
            present.append(n)
    if want is not None:
        missing = [n for n in want if n not in present]
        if missing:
            print(f"warning: no data for {missing}; plotting available only",
                  flush=True)
    # warn if series files missing (scalar plots / runtime still work)
    for n in present:
        seeds = [r["seed"] for r in rows if r["name"] == n]
        for s in seeds:
            p = os.path.join(series_root, f"{n}_seed{s}.npz")
            if not os.path.isfile(p):
                print(f"warning: missing series {p}", flush=True)
                break
    return present


def load_series(series_root, name, seed):
    path = os.path.join(series_root, f"{name}_seed{seed}.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in SERIES_KEYS if k in data}


def series_of(rows, series_root, name, key):
    out = []
    for r in rows:
        if r["name"] != name:
            continue
        ser = load_series(series_root, name, r["seed"])
        if ser is None or key not in ser:
            continue
        out.append(np.asarray(ser[key], dtype=float))
    return out


def _stack_mean_std(series_list):
    """Stack seed curves to [nseed, T]; return mean, std, raw (NaN-aware)."""
    if not series_list:
        return None, None, None
    L = min(len(s) for s in series_list)
    arr = np.stack([s[:L] for s in series_list], axis=0).astype(np.float64)
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        std = np.nanstd(arr, axis=0, ddof=1)
    else:
        std = np.zeros(L)
    return mean, std, arr


def export_mat(path, rows, series_root, names, slide):
    """Write MATLAB-friendly plot_data.mat.

    MATLAB usage::

        S = load('plot_data.mat');
        % S.algo_names{i}, S.labels{i}
        % S.summary.reward(i), S.summary.decision_ms(i), ...
        % S.series.causal.cum_reward_mean  % [T x 1]
        % S.series.causal.cum_reward       % [nseed x T] raw
        % S.series.causal.seeds            % [nseed x 1]
    """
    # Runtime fields: median across seeds (mean is dominated by rare wall outliers,
    # e.g. one PPO seed with update_ms≈900). Quality metrics stay as mean.
    _RUNTIME_MEDIAN = frozenset({
        "ms_per_slot", "decision_ms", "select_ms", "comm_ms", "bcd_ms",
        "bcd_ms_wall", "update_ms", "sec",
    })
    n = len(names)
    summary = {m: np.full(n, np.nan, dtype=np.float64) for m in SCALAR_METRICS}
    for i, name in enumerate(names):
        sub = [r for r in rows if r["name"] == name]
        if not sub:
            continue
        for m in SCALAR_METRICS:
            vals = [r[m] for r in sub if m in r]
            if not vals:
                continue
            if m in _RUNTIME_MEDIAN:
                summary[m][i] = float(np.median(vals))
            else:
                summary[m][i] = float(np.mean(vals))

    series_struct = {}
    for name in names:
        entry = {
            "seeds": np.asarray(
                [r["seed"] for r in rows if r["name"] == name], dtype=np.int64),
        }
        for key in SERIES_KEYS:
            mean, std, raw = _stack_mean_std(series_of(rows, series_root, name, key))
            if raw is None:
                continue
            entry[key] = raw                      # [nseed, T]
            entry[f"{key}_mean"] = mean.reshape(-1, 1)
            entry[f"{key}_std"] = std.reshape(-1, 1)
            if key == "rew_series" and slide > 1 and len(mean) >= slide:
                sm = sliding_mean(mean, slide)
                # per-seed sliding then mean/std
                slid = np.stack([sliding_mean(raw[i], slide) for i in range(raw.shape[0])])
                entry["rew_sliding"] = slid
                entry["rew_sliding_mean"] = slid.mean(axis=0).reshape(-1, 1)
                entry["rew_sliding_std"] = (
                    slid.std(axis=0, ddof=1).reshape(-1, 1) if slid.shape[0] > 1
                    else np.zeros((slid.shape[1], 1)))
                entry["rew_sliding_t"] = (
                    np.arange(len(sm), dtype=np.float64) + slide).reshape(-1, 1)
        series_struct[name] = entry

    payload = {
        "algo_names": np.array(names, dtype=object),
        "labels": np.array([LABELS.get(n, n) for n in names], dtype=object),
        "slide": np.int32(slide),
        "summary": summary,
        "series": series_struct,
        # flat runtime vectors (same order as algo_names) for bar plots
        "decision_ms": summary["decision_ms"],
        "comm_ms": summary["comm_ms"],
        "bcd_ms": summary["bcd_ms"],
        "update_ms": summary["update_ms"],
        "ms_per_slot": summary["ms_per_slot"],
        "comm_uplink_B": summary["comm_uplink_B"],
        "comm_downlink_B": summary["comm_downlink_B"],
        "comm_rounds": summary["comm_rounds"],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_mat_and_python(path, payload, long_field_names=True)
    return path


def plot_results(indir, out_dir=None, algos=None, slide=5, mat_path=None,
                 no_mat=False, user_num=None):
    out_dir = out_dir or indir
    os.makedirs(out_dir, exist_ok=True)
    rows = load_perseed(indir)
    rows, comm_cfg = enrich_runtime_rows(rows, user_num=user_num)
    sroot = series_dir(indir)
    want = set(algos) if algos else None
    names = discover_algos(rows, sroot, want)
    if not names:
        raise SystemExit("no algorithms to plot")

    print(f"plotting: {names}", flush=True)
    print(f"comm model: users={comm_cfg['user_num']} "
          f"RTT={comm_cfg['rtt_s']*1e3:.2g}ms "
          f"R_ctrl={comm_cfg['ctrl_rate_bps']:.3g}bps", flush=True)
    bcd_meds = []
    for n in names:
        vs = [r["bcd_ms"] for r in rows if r["name"] == n and "bcd_ms" in r]
        if vs:
            bcd_meds.append(f"{n}={float(np.median(vs)):.2f}")
    if bcd_meds:
        print(f"resource allocation (per-algo median ms/slot): "
              f"{', '.join(bcd_meds)}",
              flush=True)

    # Trailing-window reward; ylim from post-warmup percentiles.
    rew_w = 60
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_curve = False
    for name in names:
        s = series_of(rows, sroot, name, "rew_series")
        if s:
            s0 = [np.nan_to_num(np.asarray(v, dtype=float), nan=0.0) for v in s]
            s_avg = [windowed_running_mean(v, rew_w) for v in s0]
            bandplot(ax, s_avg, COLORS.get(name, "#333"), LABELS.get(name, name),
                     robust=False, band=True, alpha=0.12, lw=1.9)
            any_curve = True
    if any_curve:
        ax.set_xlabel("Time slot")
        ax.set_ylabel("Running average reward")
        ax.set_title("Per-slot reward (Lyapunov objective)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        _autoscale_ylim(ax, pad_frac=0.12, abs_pad=2.0, q_lo=5.0, q_hi=95.0,
                        skip_first=rew_w, skip_frac=0.10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "reward.png"), dpi=160)
    plt.close(fig)
    # Drop obsolete separate sliding figure if present.
    old_slide = os.path.join(out_dir, "reward_sliding.png")
    if os.path.isfile(old_slide):
        try:
            os.remove(old_slide)
        except OSError:
            pass

    components = [
        ("acc_series", "accuracy.png", "Mean accuracy (mAP)", "Accuracy", True),
        ("delay_series", "delay.png", "Mean delay [s]", "Delay", True),
        ("energy_series", "energy.png", "Mean energy [J]", "Energy", True),
        ("backlog_series", "backlog.png", "Mean backlog [tasks]", "Composite task backlog", False),
        ("vio_series", "violation.png", "Violation rate", "Constraint violation", True),
    ]
    for key, fname, ylabel, title, do_slide in components:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        any_curve = False
        # Delay/energy/backlog: same outlier-seed problem → robust bands.
        robust = key in ("delay_series", "energy_series", "backlog_series")
        for name in names:
            s = series_of(rows, sroot, name, key)
            if s:
                bandplot(ax, s, COLORS.get(name, "#333"), LABELS.get(name, name),
                         sliding=slide if do_slide else None, robust=robust)
                any_curve = True
        if any_curve:
            ylab = ylabel.replace("Mean ", "Median " if robust else "Mean ")
            ax.set_xlabel("Time slot"); ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            if robust:
                _autoscale_ylim(ax)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)

    # Runtime stacked bar: median across seeds (robust to wall-clock outliers).
    # Previously used mean → one UCB seed with decision_ms≈80ms dominated the bar.
    def _seed_vals(name, key):
        return np.asarray(
            [r[key] for r in rows if r["name"] == name and key in r],
            dtype=float)

    stack_med = {key: [] for key in RUNTIME_STACK}
    for n in names:
        for key in RUNTIME_STACK:
            v = _seed_vals(n, key)
            if v.size == 0:
                stack_med[key].append(0.0)
                continue
            stack_med[key].append(float(np.median(v)))
            if key in ("select_ms", "update_ms"):
                # flag wall-clock contamination / outliers
                med = float(np.median(v))
                mx = float(np.max(v))
                if med > 1e-9 and mx > 3.0 * med:
                    print(f"warning: {n}.{key} max/median="
                          f"{mx/med:.1f}x (max={mx:.2f}, median={med:.2f}); "
                          f"runtime bar uses median", flush=True)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottom = np.zeros(len(names), dtype=float)
    for key, label, color in zip(RUNTIME_STACK, RUNTIME_STACK_LABELS,
                                 RUNTIME_STACK_COLORS):
        vals = np.asarray(stack_med[key], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=18, ha="right")
    ax.set_ylabel("Time per slot [ms]")
    ax.set_title(
        "Runtime per slot (median across seeds)\n"
        "resource allocation = measured wall; interaction = control-plane model")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    runtime_path = os.path.join(out_dir, "runtime.png")
    fig.savefig(runtime_path, dpi=160)
    plt.close(fig)

    # Compute-only stack (no modeled interaction): selection / update / alloc.
    compute_keys = ("select_ms", "update_ms", "bcd_ms")
    compute_labels = ("selection", "update", "resource allocation")
    compute_colors = ("#4c78a8", "#9ecae9", "#f58518")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottom = np.zeros(len(names), dtype=float)
    for key, label, color in zip(compute_keys, compute_labels, compute_colors):
        vals = np.asarray(stack_med[key], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=18, ha="right")
    ax.set_ylabel("Time per slot [ms]")
    ax.set_title(
        "Compute + resource allocation per slot (median across seeds)\n"
        "excludes modeled control-plane interaction")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    runtime_compute_path = os.path.join(out_dir, "runtime_compute.png")
    fig.savefig(runtime_compute_path, dpi=160)
    plt.close(fig)

    # Drop obsolete log-scale figure if present from older plot runs
    old_log = os.path.join(out_dir, "runtime_log.png")
    if os.path.isfile(old_log):
        try:
            os.remove(old_log)
        except OSError:
            pass

    # Per-metric mean bar charts (across seeds; error bars = seed std)
    bar_specs = [
        ("reward", "reward_bar.png", "Mean Lyapunov reward", "Reward (mean ± std)"),
        ("acc", "accuracy_bar.png", "Mean accuracy (mAP)", "Accuracy (mean ± std)"),
        ("delay", "delay_bar.png", "Mean delay [s]", "Delay (mean ± std)"),
        ("energy", "energy_bar.png", "Mean energy [J]", "Energy (mean ± std)"),
        ("backlog", "backlog_bar.png", "Mean backlog [tasks]", "Composite task backlog (mean ± std)"),
        ("vio", "violation_bar.png", "Violation rate", "Constraint violation (mean ± std)"),
    ]
    bar_paths = []
    for field, fname, ylabel, title in bar_specs:
        means, stds = [], []
        for name in names:
            sub = [r for r in rows if r["name"] == name and field in r]
            if not sub:
                means.append(np.nan)
                stds.append(0.0)
                continue
            vals = np.asarray([r[field] for r in sub], dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        if np.all(np.isnan(means)):
            continue
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        colors = [COLORS.get(n, "#333") for n in names]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3,
               error_kw={"elinewidth": 1.0, "capthick": 1.0})
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        bar_paths.append(path)

    print(f"wrote plots -> {out_dir}/")
    print(f"runtime stack -> {runtime_path}", flush=True)
    print(f"runtime compute -> {runtime_compute_path}", flush=True)
    if bar_paths:
        print(f"mean bars -> {len(bar_paths)} files (*_bar.png)", flush=True)

    if not no_mat:
        mat = mat_path or os.path.join(out_dir, "plot_data.mat")
        export_mat(mat, rows, sroot, names, slide)
        print(f"wrote plot exports -> {mat} (+ .pkl/.npz)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot OMNIS train results")
    p.add_argument("--indir", default=None,
                   help="directory with perseed.csv and series/ "
                        "(default: PYCHARM_INDIR / figures/python figures/convergence)")
    p.add_argument("--out", default=None,
                   help="plot output dir (default: same as --indir)")
    p.add_argument("--algos", nargs="+", default=None,
                   help="subset to plot (default: all present in CSV)")
    p.add_argument("--slide", type=int, default=None)
    p.add_argument("--mat", default=None,
                   help="output .mat path (default: <out>/plot_data.mat)")
    p.add_argument("--no-mat", action="store_true",
                   help="skip writing plot_data.mat/.pkl/.npz")
    p.add_argument("--users", type=int, default=None,
                   help="user count for deriving comm_ms on old CSVs "
                        "(default: Config.user_num)")
    args = p.parse_args(argv)
    indir = args.indir if args.indir is not None else PYCHARM_INDIR
    out = args.out if args.out is not None else PYCHARM_OUT
    algos = args.algos if args.algos is not None else PYCHARM_ALGOS
    if algos == "all" or algos == ["all"]:
        algos = None  # plot whatever is present in CSV
    slide = args.slide if args.slide is not None else PYCHARM_SLIDE
    users = args.users if args.users is not None else PYCHARM_USERS
    no_mat = bool(args.no_mat) or (not PYCHARM_WRITE_MAT)
    plot_results(indir, out_dir=out, algos=algos, slide=slide,
                 mat_path=args.mat, no_mat=no_mat, user_num=users)


if __name__ == "__main__":
    main()
