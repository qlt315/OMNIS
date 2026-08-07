#!/usr/bin/env python3
"""Plot MAB ablation + Causal prediction-error figures.

Reads ``perseed.csv`` + ``series/*.npz`` under ``--indir`` (default
``figures/ablation``). Writes:

  - pred_error.png          Causal |acc-prior| / |acc-posterior| (+ RMSE)
  - pred_error_reward.png   optional UCB-family reward GP abs error
  - reward.png / backlog.png  base vs noupdate (MAB only; freeze hidden by default)

Usage:
  PYTHONPATH=. python3 experiments/plot_ablation.py --indir figures/ablation
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lib import MAB_BASE_ALGOS, series_dir  # noqa: E402

# Distinct styles for ablation suffixes
VARIANT_STYLES = {
    "": {"ls": "-", "lw": 2.0},
    "_noupdate": {"ls": "--", "lw": 1.8},
    "_freeze": {"ls": ":", "lw": 1.8},
}
BASE_COLORS = {
    "causal": "#1f77b4",
    "ucb": "#ff7f0e",
    "dts": "#8c564b",
    "cto": "#17becf",
}
BASE_LABELS = {
    "causal": "Causal",
    "ucb": "UCB",
    "dts": "DTS",
    "cto": "CTO",
}
SUFFIX_LABELS = {
    "": "",
    "_noupdate": " (no-update)",
    "_freeze": " (freeze)",
}


def _split_label(name):
    for suf in ("_noupdate", "_freeze"):
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def load_perseed(indir):
    path = os.path.join(indir, "perseed.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_series(indir, name, key):
    """Load list of 1-D arrays for ``name`` across seeds for series key."""
    d = series_dir(indir)
    paths = sorted(glob.glob(os.path.join(d, f"{name}_seed*.npz")))
    out = []
    for p in paths:
        with np.load(p) as z:
            if key in z.files:
                out.append(np.asarray(z[key], dtype=float))
    return out


def bandplot(ax, series_list, color, label, ls="-", lw=1.8, alpha=0.12, sliding=None):
    if not series_list:
        return
    L = min(len(s) for s in series_list)
    arr = np.stack([s[:L] for s in series_list], axis=0)
    if sliding and L >= sliding:
        kernel = np.ones(sliding) / sliding
        arr = np.stack([np.convolve(row, kernel, mode="valid") for row in arr], axis=0)
        x = np.arange(arr.shape[1]) + sliding - 1
    else:
        x = np.arange(arr.shape[1])
    m = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=alpha)


def discover_names(indir):
    rows = load_perseed(indir)
    names = []
    for r in rows:
        n = r["name"]
        if n not in names:
            names.append(n)
    if names:
        return names
    d = series_dir(indir)
    for p in sorted(glob.glob(os.path.join(d, "*_seed*.npz"))):
        base = os.path.basename(p).rsplit("_seed", 1)[0]
        if base not in names:
            names.append(base)
    return names


def plot_pred_error(indir, outpath, sliding=5, show_freeze=False):
    """Causal accuracy prediction error vs slot (mean ± std across seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    plotted = False
    for name in discover_names(indir):
        base, suf = _split_label(name)
        if base != "causal":
            continue
        if suf == "_freeze" and not show_freeze:
            continue
        color = BASE_COLORS["causal"]
        style = VARIANT_STYLES.get(suf, VARIANT_STYLES[""])
        label = BASE_LABELS["causal"] + SUFFIX_LABELS.get(suf, suf)
        prior = load_series(indir, name, "pred_err_prior")
        post = load_series(indir, name, "pred_err_post")
        rmse = load_series(indir, name, "pred_err_rmse")
        # Base: show prior vs posterior; ablations: posterior only (cleaner).
        if suf == "" and prior:
            bandplot(axes[0], prior, "#9ecae1", "Causal |acc−prior|",
                     ls="--", lw=1.6, sliding=sliding, alpha=0.1)
            plotted = True
        if post:
            bandplot(axes[0], post, color, f"{label} |acc−post|",
                     ls=style["ls"], lw=style["lw"], sliding=sliding)
            plotted = True
        if rmse:
            bandplot(axes[1], rmse, color, f"{label} RMSE",
                     ls=style["ls"], lw=style["lw"])
            plotted = True

    axes[0].set_ylabel("|acc − mean|")
    axes[0].set_xlabel("slot")
    axes[0].set_title("Causal prediction error")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("rolling RMSE")
    axes[1].set_xlabel("slot")
    axes[1].set_title("Causal rolling RMSE (posterior)")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if plotted:
        fig.savefig(outpath, dpi=160)
        print(f"wrote {outpath}")
    else:
        print(f"skip {outpath} (no Causal pred_err series)")
    plt.close(fig)


def plot_reward_pred_error(indir, outpath, sliding=5):
    """Optional UCB/DTS/CTO reward GP abs error (base variants only by default)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    plotted = False
    for name in discover_names(indir):
        base, suf = _split_label(name)
        if base not in ("ucb", "dts", "cto") or suf != "":
            continue
        series = load_series(indir, name, "pred_err_reward")
        if not series:
            continue
        # drop all-NaN early slots for cleaner mean
        cleaned = []
        for s in series:
            s = np.asarray(s, dtype=float)
            if np.all(np.isnan(s)):
                continue
            cleaned.append(s)
        if not cleaned:
            continue
        bandplot(ax, cleaned, BASE_COLORS[base], BASE_LABELS[base],
                 sliding=sliding)
        plotted = True
    ax.set_xlabel("slot")
    ax.set_ylabel("|reward − GP mean|")
    ax.set_title("MAB reward prediction error (base)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if plotted:
        fig.savefig(outpath, dpi=160)
        print(f"wrote {outpath}")
    else:
        print(f"skip {outpath} (no reward pred_err series)")
    plt.close(fig)


def plot_metric_ablation(indir, metric_key, ylabel, outpath, title, sliding=5,
                         show_freeze=False):
    """Compare base / noupdate for each MAB base on a time series."""
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    plotted = False
    for name in discover_names(indir):
        base, suf = _split_label(name)
        if base not in MAB_BASE_ALGOS:
            continue
        if suf == "_freeze" and not show_freeze:
            continue
        series = load_series(indir, name, metric_key)
        if not series:
            continue
        color = BASE_COLORS.get(base, "#333333")
        style = VARIANT_STYLES.get(suf, VARIANT_STYLES[""])
        label = BASE_LABELS.get(base, base) + SUFFIX_LABELS.get(suf, suf)
        bandplot(ax, series, color, label, ls=style["ls"], lw=style["lw"],
                 sliding=sliding)
        plotted = True
    ax.set_xlabel("slot")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    if metric_key == "backlog_series":
        ax.set_yscale("log")
    fig.tight_layout()
    if plotted:
        fig.savefig(outpath, dpi=160)
        print(f"wrote {outpath}")
    else:
        print(f"skip {outpath}")
    plt.close(fig)


def plot_bar_summary(indir, outpath, show_freeze=False):
    """Bar chart of mean reward (from perseed) for MAB ablation labels."""
    rows = load_perseed(indir)
    if not rows:
        print(f"skip {outpath} (no perseed.csv)")
        return
    agg = {}
    for r in rows:
        name = r["name"]
        base, suf = _split_label(name)
        if base not in MAB_BASE_ALGOS:
            continue
        if suf == "_freeze" and not show_freeze:
            continue
        agg.setdefault(name, []).append(float(r["reward"]))
    if not agg:
        print(f"skip {outpath}")
        return
    # order: per base then variants (noupdate only by default)
    order = []
    for base in MAB_BASE_ALGOS:
        for suf in ("", "_noupdate") + (("_freeze",) if show_freeze else ()):
            lab = f"{base}{suf}"
            if lab in agg:
                order.append(lab)
    means = [np.mean(agg[n]) for n in order]
    stds = [np.std(agg[n], ddof=1) if len(agg[n]) > 1 else 0.0 for n in order]
    colors = [BASE_COLORS[_split_label(n)[0]] for n in order]
    labels = [BASE_LABELS[_split_label(n)[0]] + SUFFIX_LABELS[_split_label(n)[1]]
              for n in order]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(order)), 4.0))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean reward")
    ax.set_title("MAB ablation: mean reward")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    print(f"wrote {outpath}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot MAB ablation + pred-error figures")
    p.add_argument("--indir", default="figures/ablation")
    p.add_argument("--sliding", type=int, default=5)
    p.add_argument("--show-freeze", action="store_true",
                   help="include freeze@K variants (hidden by default)")
    args = p.parse_args()
    indir = args.indir
    os.makedirs(indir, exist_ok=True)
    show_frz = bool(args.show_freeze)

    plot_pred_error(indir, os.path.join(indir, "pred_error.png"),
                    sliding=args.sliding, show_freeze=show_frz)
    plot_reward_pred_error(indir, os.path.join(indir, "pred_error_reward.png"),
                           sliding=args.sliding)
    plot_metric_ablation(
        indir, "rew_series", "Lyapunov reward",
        os.path.join(indir, "reward.png"),
        "MAB ablation: reward vs slot (base vs no-update)",
        sliding=args.sliding, show_freeze=show_frz)
    plot_metric_ablation(
        indir, "backlog_series", "backlog [bits]",
        os.path.join(indir, "backlog.png"),
        "MAB ablation: backlog vs slot (base vs no-update)",
        sliding=args.sliding, show_freeze=show_frz)
    plot_bar_summary(indir, os.path.join(indir, "reward_bar.png"),
                     show_freeze=show_frz)


if __name__ == "__main__":
    main()
