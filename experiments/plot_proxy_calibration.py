#!/usr/bin/env python3
"""Verify admission-proxy accuracy vs slot (MD / ES panels).

X-axis: time slot. Y-axis: normalized absolute estimation error
``|hat - real| / max(|real|, eps)``, averaged over tasks that finish in
that slot, then smoothed with a trailing window (same style as reward
convergence plots).

Produces two figures:
  * MD-side: local, uplink, total delay/energy
  * ES-side: edge, service, total delay/energy

Example
-------
  PYTHONPATH=. python3 experiments/plot_proxy_calibration.py \\
      --algo causal --slots 250 --users 15 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from repo_util import ensure_repo_root

ensure_repo_root()

from experiments.convergence_lib import ALGOS, configure
from experiments.result_io import save_mat_and_python

# MD-side: local + admit uplink (scaled formula) + totals.
MD_METRICS = [
    ("Local delay", "hat_local_d", "real_local_d", "#1e3a5f"),
    ("Local energy", "hat_local_e", "real_local_e", "#5b9bd5"),
    ("Uplink delay", "hat_tx_d", "real_tx_d", "#6e2c00"),
    ("Uplink energy", "hat_tx_e", "real_tx_e", "#e67e22"),
    ("Total delay", "hat_sojourn_d", "real_e2e_d", "#4a235a"),
    ("Total energy", "hat_total_e", "real_total_e", "#bb8fce"),
]

# ES-side: edge EWMA + composites with measured uplink (timestamps).
ES_METRICS = [
    ("Edge delay", "hat_edge_d", "real_edge_d", "#145a32"),
    ("Edge energy", "hat_edge_e", "real_edge_e", "#58d68d"),
    ("Service delay", "hat_service_d_es", "real_service_d", "#1a5276"),
    ("Total delay", "hat_sojourn_d_es", "real_e2e_d", "#4a235a"),
    ("Total energy", "hat_total_e_es", "real_total_e", "#bb8fce"),
]

# Backward-compatible flat list (summary / mat dump).
METRICS = MD_METRICS + [
    m for m in ES_METRICS if m[1] not in {x[1] for x in MD_METRICS}
]


def _algo_cls(name: str):
    for n, cls in ALGOS:
        if n == name:
            return cls
    raise SystemExit(f"unknown algo {name!r}; choose from {[n for n, _ in ALGOS]}")


def _norm_err(hat, real, eps: float = 1e-9) -> float:
    h = float(hat)
    r = float(real)
    if not (np.isfinite(h) and np.isfinite(r)):
        return float("nan")
    return abs(h - r) / max(abs(r), eps)


def _slot_of_row(row) -> int:
    if "slot" in row and row["slot"] is not None and str(row["slot"]) != "":
        try:
            return int(float(row["slot"]))
        except (TypeError, ValueError):
            pass
    for key in ("t_done", "t_admit"):
        if key in row and row[key] is not None and str(row[key]) != "":
            try:
                t = float(row[key])
                if np.isfinite(t):
                    return max(0, int(t))
            except (TypeError, ValueError):
                continue
    return -1


def per_slot_norm_error(rows, n_slots: int, hat_key: str, real_key: str,
                        reduce: str = "median"):
    """Normalized |hat-real|/|real| per completion slot (median or mean)."""
    buckets = [[] for _ in range(int(n_slots))]
    for row in rows:
        s = _slot_of_row(row)
        if s < 0 or s >= n_slots:
            continue
        if hat_key not in row or real_key not in row:
            continue
        err = _norm_err(row[hat_key], row[real_key])
        if np.isfinite(err):
            buckets[s].append(err)
    series = np.full(n_slots, np.nan, dtype=float)
    for s, vals in enumerate(buckets):
        if not vals:
            continue
        if reduce == "mean":
            series[s] = float(np.mean(vals))
        else:
            series[s] = float(np.median(vals))
    return series


def windowed_running_mean(x, w: int):
    """Trailing-window mean; NaNs skipped (same idea as plot_results)."""
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


def rows_to_arrays(rows):
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k not in ("user", "model")]
    out = {
        "user": np.array([r["user"] for r in rows], dtype=object),
        "model": np.array([r["model"] for r in rows], dtype=object),
    }
    for k in keys:
        try:
            out[k] = np.asarray([float(r[k]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            out[k] = np.array([r[k] for r in rows], dtype=object)
    return out


def _stats(hat, real):
    hat = np.asarray(hat, dtype=float)
    real = np.asarray(real, dtype=float)
    m = np.isfinite(hat) & np.isfinite(real)
    if not np.any(m):
        return {"n": 0, "mae": np.nan, "mape": np.nan, "corr": np.nan, "bias": np.nan}
    h, r = hat[m], real[m]
    mae = float(np.mean(np.abs(h - r)))
    denom = np.maximum(np.abs(r), 1e-9)
    mape = float(np.mean(np.abs(h - r) / denom))
    bias = float(np.mean(h - r))
    if h.size >= 2 and np.std(h) > 1e-12 and np.std(r) > 1e-12:
        corr = float(np.corrcoef(h, r)[0, 1])
    else:
        corr = float("nan")
    return {"n": int(h.size), "mae": mae, "mape": mape, "corr": corr, "bias": bias}


def _plot_one_panel(rows, n_slots, metrics, out_path, title, window=20, log_y=True):
    """Draw one error-vs-slot panel; return (series_raw, series_smooth)."""
    series_raw = {}
    series_smooth = {}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(int(n_slots))
    for label, hk, rk, color in metrics:
        if not rows or hk not in rows[0]:
            continue
        raw = per_slot_norm_error(rows, n_slots, hk, rk, reduce="median")
        smooth = windowed_running_mean(raw, window)
        series_raw[hk] = raw
        series_smooth[hk] = smooth
        y = 100.0 * smooth
        if log_y:
            y = np.where(np.isfinite(y), np.clip(y, 1e-2, None), np.nan)
        ax.plot(x, y, color=color, lw=1.9, label=label)

    ax.set_xlabel("Time slot")
    ax.set_ylabel("Normalized estimation error [%]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(0, max(n_slots - 1, 0))
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e3)
    else:
        finite = []
        for s in series_smooth.values():
            m = np.isfinite(s)
            if m.any():
                finite.append(100.0 * s[m])
        if finite:
            all_y = np.concatenate(finite)
            hi = float(np.percentile(all_y, 95))
            ax.set_ylim(0.0, max(min(hi * 1.15, 200.0), 20.0))
        else:
            ax.set_ylim(0.0, 100.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return series_raw, series_smooth


def plot_error_vs_slot(rows, n_slots: int, out_dir: str, stem="proxy_calibration",
                       window: int = 20, log_y: bool = True):
    """Two figures: MD-side hats and ES-side hats (normalized error [%] vs slot)."""
    os.makedirs(out_dir, exist_ok=True)
    arr = rows_to_arrays(rows)
    summary = {}
    for label, hk, rk, _color in METRICS:
        del label
        if hk in arr and rk in arr:
            summary[hk] = _stats(arr[hk], arr[rk])
        else:
            summary[hk] = _stats([], [])

    title_sfx = f"(per-slot median, trailing window = {window})"
    md_png = os.path.join(out_dir, f"{stem}_md_error_vs_slot.png")
    es_png = os.path.join(out_dir, f"{stem}_es_error_vs_slot.png")
    md_raw, md_smooth = _plot_one_panel(
        rows, n_slots, MD_METRICS, md_png,
        f"MD admission-proxy error vs slot {title_sfx}",
        window=window, log_y=log_y)
    es_raw, es_smooth = _plot_one_panel(
        rows, n_slots, ES_METRICS, es_png,
        f"ES admission-proxy error vs slot {title_sfx}",
        window=window, log_y=log_y)

    series_raw = {**md_raw, **es_raw}
    series_smooth = {**md_smooth, **es_smooth}
    return [md_png, es_png], summary, arr, series_raw, series_smooth


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run(algo="causal", seed=0, slots=200, users=15, out_dir=None, window=20,
        from_csv=None, log_y=True):
    out_dir = out_dir or os.path.join(
        "figures", "python figures", "proxy_calibration")
    os.makedirs(out_dir, exist_ok=True)

    if from_csv:
        rows = load_csv(from_csv)
        print(f"loaded {len(rows)} rows from {from_csv}", flush=True)
        n_slots = int(slots)
        if rows:
            done = [float(r["t_done"]) for r in rows
                    if r.get("t_done") not in (None, "")]
            if done:
                n_slots = max(n_slots, int(max(done)) + 1)
    else:
        cls = _algo_cls(algo)
        c = configure(algo, seed, slots, users, log_pred_error=False)
        c.log_proxy_calib = True
        agent = cls(c)
        agent.log_proxy_calib = True
        if not hasattr(agent, "proxy_calib"):
            agent.proxy_calib = []
        print(f"running {algo} seed={seed} slots={slots} users={users} ...",
              flush=True)
        agent.simulation()
        rows = list(getattr(agent, "proxy_calib", []) or [])
        n_slots = int(slots)
        csv_path = os.path.join(out_dir, "proxy_calib_per_task.csv")
        write_csv(rows, csv_path)
        print(f"wrote {csv_path}")

    print(f"completed tasks with hats: {len(rows)}", flush=True)
    if not rows:
        raise SystemExit("no calibration rows; check log_proxy_calib hook")

    pngs, summary, arr, series_raw, series_smooth = plot_error_vs_slot(
        rows, n_slots, out_dir, window=window, log_y=log_y)

    mat_payload = {
        "algo": np.array([algo], dtype=object),
        "seed": np.array([seed], dtype=int),
        "slots": np.array([n_slots], dtype=int),
        "users": np.array([users], dtype=int),
        "n_tasks": np.array([len(rows)], dtype=int),
        "window": np.array([window], dtype=int),
        **arr,
        "summary_mae": np.array(
            [summary[k]["mae"] for k in sorted(summary)], dtype=float),
        "summary_mape": np.array(
            [summary[k]["mape"] for k in sorted(summary)], dtype=float),
        "summary_corr": np.array(
            [summary[k]["corr"] for k in sorted(summary)], dtype=float),
        "summary_keys": np.array(sorted(summary), dtype=object),
    }
    for hk, raw in series_raw.items():
        mat_payload[f"err_raw_{hk}"] = raw
        mat_payload[f"err_smooth_{hk}"] = series_smooth[hk]
    mat_path, pkl_path, npz_path = save_mat_and_python(
        os.path.join(out_dir, "proxy_calibration.mat"), mat_payload)

    sum_csv = os.path.join(out_dir, "proxy_calib_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mae", "mape", "corr", "bias"])
        for k in sorted(summary):
            st = summary[k]
            w.writerow([k, st["n"], st["mae"], st["mape"], st["corr"], st["bias"]])

    for png in pngs:
        print(f"wrote {png}")
    print(f"wrote {mat_path}")
    if pkl_path:
        print(f"wrote {pkl_path}")
    if npz_path:
        print(f"wrote {npz_path}")
    print(f"wrote {sum_csv}")
    for k in sorted(summary):
        st = summary[k]
        print(f"  {k}: MAPE={st['mape']:.2%}  MAE={st['mae']:.3g}  "
              f"r={st['corr']:.3f}  n={st['n']}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", default="causal",
                   choices=[n for n, _ in ALGOS])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slots", type=int, default=250)
    p.add_argument("--users", type=int, default=15)
    p.add_argument("--window", type=int, default=20,
                   help="trailing window for smoothed error curves")
    p.add_argument("--from-csv", default=None,
                   help="replot from an existing proxy_calib_per_task.csv")
    p.add_argument("--linear-y", action="store_true", default=True,
                   help="linear y-axis in percent (default)")
    p.add_argument("--log-y", action="store_true",
                   help="use log y-axis instead of linear percent")
    p.add_argument("--out", default="figures/python figures/proxy_calibration")
    args = p.parse_args()
    run(algo=args.algo, seed=args.seed, slots=args.slots,
        users=args.users, out_dir=args.out, window=args.window,
        from_csv=args.from_csv, log_y=bool(args.log_y))


if __name__ == "__main__":
    main()
