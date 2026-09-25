#!/usr/bin/env python3
"""Acc–vio Pareto: fixed env; OMNIS+ / GDO sweep feas_margin."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
sys.path.insert(0, _ROOT)

from experiments.convergence_lib import ALGOS  # noqa: E402
from experiments.result_io import save_mat_and_python  # noqa: E402
from experiments.sweep_lib import (  # noqa: E402
    COLORS, DEFAULT_SLOTS, offset_for_snr_target, run_one_sweep,
    sweep_outdir,
)

ALGOS_PARETO = ("causal", "gdo")
LABEL = {"causal": "OMNIS+", "gdo": "GDO"}
KNOB_COL = {
    "causal": "causal_feas_margin",
    "gdo": "gdo_feas_margin",
}
MARKERS = {"causal": "s", "gdo": "o"}
ANNOT_FONTSIZE = 11


def _agg(rows, name):
    key = KNOB_COL[name]
    knobs = sorted({float(r[key]) for r in rows
                    if r["name"] == name and np.isfinite(float(r[key]))})
    out = []
    for k in knobs:
        sub = [r for r in rows
               if r["name"] == name and abs(float(r[key]) - k) < 1e-12]
        if not sub:
            continue
        acc = np.asarray([float(r["acc"]) for r in sub])
        vio = np.asarray([float(r["vio"]) for r in sub])
        rew = np.asarray([float(r["reward"]) for r in sub])
        out.append({
            "knob": k,
            "acc_m": float(acc.mean()), "acc_s": float(acc.std()),
            "vio_m": float(vio.mean()), "vio_s": float(vio.std()),
            "rew_m": float(rew.mean()), "rew_s": float(rew.std()),
        })
    out.sort(key=lambda d: d["vio_m"])
    return out


def _interp_acc_at_vio(curve, target_vio):
    if len(curve) < 2:
        return None
    v = np.asarray([c["vio_m"] for c in curve], dtype=float)
    a = np.asarray([c["acc_m"] for c in curve], dtype=float)
    order = np.argsort(v)
    v, a = v[order], a[order]
    keep = np.ones(len(v), dtype=bool)
    for i in range(1, len(v)):
        if abs(v[i] - v[i - 1]) < 1e-4:
            keep[i - 1] = False
    v, a = v[keep], a[keep]
    if len(v) < 2 or target_vio < v[0] - 1e-6 or target_vio > v[-1] + 1e-6:
        return None
    return float(np.interp(target_vio, v, a))


def _plot_pareto(rows, out_png, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    curves = {}
    for name in ALGOS_PARETO:
        curve = _agg(rows, name)
        curves[name] = curve
        if not curve:
            continue
        color = COLORS.get(name, "#333")
        ax.errorbar(
            [c["vio_m"] for c in curve],
            [c["acc_m"] for c in curve],
            xerr=[c["vio_s"] for c in curve],
            yerr=[c["acc_s"] for c in curve],
            fmt=MARKERS.get(name, "o") + "-", color=color, lw=2.0, ms=8,
            capsize=3, label=LABEL.get(name, name),
        )
        for c in curve:
            ax.annotate(
                f"{c['knob']:.2f}", (c["vio_m"], c["acc_m"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=ANNOT_FONTSIZE, color=color, alpha=0.9,
            )
    ax.set_xlabel("Violation probability", fontsize=12)
    ax.set_ylabel("Mean accuracy (mAP)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return curves


def _mat_payload(curves):
    payload = {
        "algo_keys": np.array(ALGOS_PARETO, dtype=object),
        "algo_labels": np.array(
            [LABEL[a] for a in ALGOS_PARETO], dtype=object),
        "knob_names": np.array(["feas_margin", "feas_margin"], dtype=object),
    }
    algos, knobs = [], []
    acc_m, acc_s, vio_m, vio_s, rew_m, rew_s = [], [], [], [], [], []
    for name in ALGOS_PARETO:
        curve = curves.get(name) or []
        kn = np.asarray([c["knob"] for c in curve], dtype=float)
        am = np.asarray([c["acc_m"] for c in curve], dtype=float)
        astd = np.asarray([c["acc_s"] for c in curve], dtype=float)
        vm = np.asarray([c["vio_m"] for c in curve], dtype=float)
        vstd = np.asarray([c["vio_s"] for c in curve], dtype=float)
        rm = np.asarray([c["rew_m"] for c in curve], dtype=float)
        rstd = np.asarray([c["rew_s"] for c in curve], dtype=float)
        payload[f"{name}_knob"] = kn
        payload[f"{name}_knob_name"] = "feas_margin"
        payload[f"{name}_acc_mean"] = am
        payload[f"{name}_acc_std"] = astd
        payload[f"{name}_vio_mean"] = vm
        payload[f"{name}_vio_std"] = vstd
        payload[f"{name}_reward_mean"] = rm
        payload[f"{name}_reward_std"] = rstd
        for c in curve:
            algos.append(name)
            knobs.append(c["knob"])
            acc_m.append(c["acc_m"]); acc_s.append(c["acc_s"])
            vio_m.append(c["vio_m"]); vio_s.append(c["vio_s"])
            rew_m.append(c["rew_m"]); rew_s.append(c["rew_s"])
    payload["algos"] = np.array(algos, dtype=object)
    payload["knobs"] = np.asarray(knobs, dtype=float)
    payload["acc_mean"] = np.asarray(acc_m, dtype=float)
    payload["acc_std"] = np.asarray(acc_s, dtype=float)
    payload["vio_mean"] = np.asarray(vio_m, dtype=float)
    payload["vio_std"] = np.asarray(vio_s, dtype=float)
    payload["reward_mean"] = np.asarray(rew_m, dtype=float)
    payload["reward_std"] = np.asarray(rew_s, dtype=float)
    return payload


def _iso_vio_table(curves, targets=None):
    ca, cb = curves.get("causal") or [], curves.get("gdo") or []
    if targets is None:
        if len(ca) < 2 or len(cb) < 2:
            return []
        lo = max(ca[0]["vio_m"], cb[0]["vio_m"])
        hi = min(ca[-1]["vio_m"], cb[-1]["vio_m"])
        if hi - lo < 0.01:
            return []
        targets = list(np.linspace(lo, hi, 7))
    rows = []
    for v in targets:
        aa = _interp_acc_at_vio(ca, v)
        bb = _interp_acc_at_vio(cb, v)
        if aa is None or bb is None:
            continue
        rows.append({
            "vio": float(v),
            "acc_causal": aa,
            "acc_gdo": bb,
            "delta_acc": aa - bb,
        })
    return rows


def _run_grid(name, cls, seeds, slots, users, off, knob_kw, values, rows):
    col = KNOB_COL[name]
    for v in values:
        for seed in seeds:
            print(f"[acc_vio/{name}] seed={seed} {col}={v}", flush=True)
            r = run_one_sweep(
                name, cls, seed, slots, users,
                sinr_offset_db=off, **{knob_kw: float(v)},
            )
            r["causal_feas_margin"] = float("nan")
            r["gdo_feas_margin"] = float("nan")
            r[col] = float(v)
            rows.append(r)
            print(f"    acc={r['acc']:.4f} vio={r['vio']:.3f} "
                  f"reward={r['reward']:.1f}", flush=True)


def _finalize(rows, out_dir):
    fields = ["name", "seed", "causal_feas_margin", "gdo_feas_margin",
              "reward", "acc", "vio", "delay", "energy", "backlog", "sec"]
    csv_path = os.path.join(out_dir, "perseed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    curves = _plot_pareto(
        rows,
        os.path.join(out_dir, "acc_vio_pareto.png"),
        "Acc–vio Pareto (shared env; feas_margin)",
    )
    mat = _mat_payload(curves)
    save_mat_and_python(os.path.join(out_dir, "acc_vio_pareto.mat"), mat)

    iso = _iso_vio_table(curves)
    with open(os.path.join(out_dir, "iso_vio.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["vio", "acc_causal", "acc_gdo", "delta_acc"])
        w.writeheader()
        for r in iso:
            w.writerow(r)
    if iso:
        print("[acc_vio] iso-vio Acc (OMNIS+ − GDO):", flush=True)
        for r in iso:
            print(f"    vio={r['vio']:.3f}  causal={r['acc_causal']:.4f}  "
                  f"gdo={r['acc_gdo']:.4f}  Δ={r['delta_acc']:+.4f}",
                  flush=True)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        vs = [r["vio"] for r in iso]
        ds = [r["delta_acc"] for r in iso]
        bar_c = ["#2ca02c" if d >= 0 else "#d62728" for d in ds]
        ax.bar([f"{v:.2f}" for v in vs], ds, color=bar_c, width=0.65)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("Matched violation probability", fontsize=12)
        ax.set_ylabel("Δ Acc (OMNIS+ − GDO)", fontsize=12)
        ax.set_title("Iso-vio accuracy gap", fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "iso_vio_delta.png"), dpi=160)
        plt.close(fig)
        mat["iso_vio"] = np.asarray([r["vio"] for r in iso], dtype=float)
        mat["iso_acc_causal"] = np.asarray(
            [r["acc_causal"] for r in iso], dtype=float)
        mat["iso_acc_gdo"] = np.asarray(
            [r["acc_gdo"] for r in iso], dtype=float)
        mat["iso_delta_acc"] = np.asarray(
            [r["delta_acc"] for r in iso], dtype=float)
        save_mat_and_python(os.path.join(out_dir, "acc_vio_pareto.mat"), mat)
    return curves


def sweep_acc_vio(
    seeds=(0, 1, 2),
    slots=DEFAULT_SLOTS,
    users=25,
    snr_db=5.0,
    causal_margins=None,
    gdo_margins=None,
    out_root="figures/python figures/sweeps",
):
    causal_margins = list(causal_margins or [0.78, 0.84, 0.88, 0.92, 0.96, 1.00])
    gdo_margins = list(gdo_margins or [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    cls_map = dict(ALGOS)
    out_dir = sweep_outdir(out_root, "acc_vio")
    off = offset_for_snr_target(snr_db, users=users)
    rows = []
    t0 = time.time()

    _run_grid("causal", cls_map["causal"], seeds, slots, users, off,
              "causal_feas_margin", causal_margins, rows)
    _run_grid("gdo", cls_map["gdo"], seeds, slots, users, off,
              "gdo_feas_margin", gdo_margins, rows)

    _finalize(rows, out_dir)
    print(f"[acc_vio] done in {(time.time()-t0)/60:.1f} min → {out_dir}/",
          flush=True)
    return rows, out_dir


def replot_from_csv(out_root="figures/python figures/sweeps"):
    """Drop non-causal/gdo rows and refresh figures/mat."""
    out_dir = sweep_outdir(out_root, "acc_vio")
    csv_path = os.path.join(out_dir, "perseed.csv")
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("name") not in ALGOS_PARETO:
                continue
            row = dict(r)
            for k in ("seed", "reward", "acc", "vio", "delay", "energy",
                      "backlog", "sec", "causal_feas_margin", "gdo_feas_margin"):
                if k not in row or row[k] in ("", None):
                    continue
                try:
                    row[k] = float(row[k]) if k != "seed" else int(float(row[k]))
                except ValueError:
                    row[k] = float("nan")
            rows.append(row)
    _finalize(rows, out_dir)
    print(f"[acc_vio] replot {len(rows)} rows → {out_dir}/", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    p.add_argument("--users", type=int, default=25)
    p.add_argument("--snr-db", type=float, default=5.0)
    p.add_argument("--causal-margins", type=float, nargs="+",
                   default=[0.78, 0.84, 0.88, 0.92, 0.96, 1.00])
    p.add_argument("--gdo-margins", type=float, nargs="+",
                   default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    p.add_argument("--replot-only", action="store_true",
                   help="Rebuild plots/mat from existing causal/gdo CSV rows")
    p.add_argument("--out-root", default="figures/python figures/sweeps")
    args = p.parse_args()
    if args.replot_only:
        replot_from_csv(out_root=args.out_root)
    else:
        sweep_acc_vio(
            seeds=tuple(args.seeds), slots=args.slots, users=args.users,
            snr_db=args.snr_db, causal_margins=args.causal_margins,
            gdo_margins=args.gdo_margins, out_root=args.out_root,
        )
