"""Uplink channel-gain traces for the 7-cell system model.

Each system slot is one second. The trace stores the narrowband MIMO
channel (paths summed) from every MD to every cell, not a SINR. SINR is
computed in the slot loop from the users who are actually transmitting
and from the bandwidth BCD gave them.

Topology
    7 omnidirectional cells at the hex-site centers (inter-site distance
    200 m). Sector maxima are not taken. Each BS is a 2×4 array with an
    omnidirectional element pattern; each MD has 2 antennas.

Time
    Pedestrian speed 0.8 m/s. Samples inside a chunk share one cluster
    realization and a continuous Doppler clock at 1 Hz. The next chunk
    moves the MDs and draws a new shadowing realization, on the order of
    the UMi shadowing decorrelation distance. MDs bounce inside the
    deployment instead of walking out of it.

Run from ``phy_sim/``::

    python run_traces.py --slots 1000 --tag smoke7
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sionna.phy.channel.tr38901 import PanelArray, UMi
from sionna.sys import gen_hexgrid_topology

P_MAX_W = 0.1
P0_DBM = -85.0
ALPHA = 0.8
PRB_HZ = 180e3
CELL_BW_HZ = 7.5e5
NOISE_FIGURE_DB = 8.0
CARRIER_FREQ = 3.5e9
# Thermal noise PSD [W/Hz] with the receiver noise figure.
NOISE_PSD = 10 ** ((-174.0 + NOISE_FIGURE_DB) / 10.0 - 3.0)


def _site_centers(bs_loc):
    """One position per hex site. ``gen_hexgrid_topology`` repeats it 3 times."""
    return bs_loc[:, 0::3, :].contiguous()


def _bounce(ut, vel, limit):
    """Keep MDs inside a disk of radius ``limit`` by reflecting the radial velocity."""
    xy = ut[..., :2]
    vxy = vel[..., :2]
    radius = torch.linalg.vector_norm(xy, dim=-1, keepdim=True)
    outside = radius > limit
    radial = xy / radius.clamp_min(1.0)
    ut_xy = torch.where(outside, radial * limit, xy)
    radial_speed = (vxy * radial).sum(dim=-1, keepdim=True)
    v_ref = vxy - 2.0 * radial_speed * radial
    vel_xy = torch.where(outside, v_ref, vxy)
    ut = ut.clone()
    vel = vel.clone()
    ut[..., :2] = ut_xy
    vel[..., :2] = vel_xy
    return ut, vel


def generate_traces(args):
    torch.manual_seed(args.seed)
    ue_ants = max(1, int(args.ue_ants))
    n_layers = max(1, min(int(args.n_layers), ue_ants, args.bs_rows * args.bs_cols))
    chunk = max(1, int(args.time_chunk))
    speed = float(args.speed)

    bs_array = PanelArray(
        num_rows_per_panel=args.bs_rows,
        num_cols_per_panel=args.bs_cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQ,
    )
    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=ue_ants,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQ,
    )
    cm = UMi(
        carrier_frequency=CARRIER_FREQ,
        o2i_model="low",
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        enable_pathloss=True,
        enable_shadow_fading=True,
    )
    # 2 UEs per sector × 3 sectors = 6 UEs per site. 7 sites → 42 UEs.
    topo = gen_hexgrid_topology(
        batch_size=1,
        num_rings=args.num_rings,
        num_ut_per_sector=args.ues_per_cell,
        scenario="umi",
        isd=args.isd,
        min_ut_velocity=speed,
        max_ut_velocity=speed,
    )
    ut, bs, ut_ori, bs_ori, vel, indoor, _los, _bs_virtual = topo
    sites = _site_centers(bs)
    n_cells = int(sites.shape[1])
    n_ut = int(ut.shape[1])
    bs_ori = bs_ori[:, 0::3, :].contiguous()
    bs_virtual = sites.unsqueeze(2).expand(-1, -1, n_ut, -1).contiguous()
    limit = 2.0 * float(args.isd)

    T = int(args.slots)
    # Filled after the first chunk reveals Nr, Nt.
    H_out = None
    rel = []
    for t0 in range(0, T, chunk):
        tc = min(chunk, T - t0)
        ut, vel = _bounce(ut, vel, limit)
        cm.set_topology(
            ut, sites, ut_ori, bs_ori, vel, indoor, bs_virtual_loc=bs_virtual)
        h, _delays = cm(tc, 1.0)
        # (1, C, Nr, U, Nt, paths, time) → sum paths → (C, U, Nr, Nt, time)
        summed = h[0].sum(dim=4)
        summed = summed.permute(0, 2, 1, 3, 4).contiguous()
        if H_out is None:
            _c, _u, n_rx, n_tx, _tc = summed.shape
            H_out = np.empty((T, n_cells, n_ut, n_rx, n_tx), dtype=np.complex64)
        block = summed.detach().cpu().numpy().astype(np.complex64, copy=False)
        H_out[t0:t0 + tc] = np.moveaxis(block, -1, 0)
        if tc > 1:
            a = block[..., 0]
            b = block[..., 1]
            rel.append(float(np.abs(a - b).mean() / max(np.abs(a).mean(), 1e-30)))
        ut = ut + vel * float(tc)
        del h, summed
        print(f"  slots {t0}:{t0 + tc}/{T}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"channel_{args.tag}.npz")
    np.savez_compressed(
        path,
        H=H_out,
        bandwidth_hz=np.float64(CELL_BW_HZ),
        noise_psd_w_hz=np.float64(NOISE_PSD),
        p_max_w=np.float64(P_MAX_W),
        p0_dbm=np.float64(P0_DBM),
        alpha=np.float64(ALPHA),
        prb_hz=np.float64(PRB_HZ),
        n_layers=np.int32(n_layers),
        carrier_hz=np.float64(CARRIER_FREQ),
        isd_m=np.float64(args.isd),
        speed_mps=np.float64(speed),
    )
    meta = {
        "phy": "channel_gain",
        "direction": "uplink",
        "scenario": "umi",
        "cells": "7 omni sites, no sector max",
        "bs_rows": args.bs_rows,
        "bs_cols": args.bs_cols,
        "n_rx": int(H_out.shape[3]),
        "ue_ants": ue_ants,
        "n_tx": int(H_out.shape[4]),
        "n_layers": n_layers,
        "rate_layers": n_layers,
        "effective_sinr": "irc-eesm",
        "eesm_beta": 1.0,
        "p_max_W": P_MAX_W,
        "p0_dBm": P0_DBM,
        "alpha": ALPHA,
        "prb_Hz": PRB_HZ,
        "bandwidth_Hz": CELL_BW_HZ,
        "noise_figure_dB": NOISE_FIGURE_DB,
        "noise_psd_W_per_Hz": NOISE_PSD,
        "speed_mps": speed,
        "slots": T,
        "slot_s": 1.0,
        "time_chunk": chunk,
        "num_cells": n_cells,
        "num_ues": n_ut,
        "seed": args.seed,
        "tag": args.tag,
        "note": (
            "H is the path-summed uplink channel per (slot, cell, UE). "
            "SINR is not stored. A chunk shares clusters and a 1 Hz Doppler "
            "clock; the next chunk moves the UEs and redraws shadowing. "
            "rate_layers is the number of spatial layers whose spectral "
            "efficiency is summed for one transport block."
        ),
    }
    meta_path = os.path.join(args.out_dir, f"channel_{args.tag}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if rel:
        print(f"mean relative |H(t+1)-H(t)| inside a chunk: {np.mean(rel):.3f}",
              flush=True)
    print(f"wrote {path}", flush=True)
    print(f"wrote {meta_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["umi", "uma"], default="umi")
    p.add_argument("--num-rings", type=int, default=1)
    p.add_argument("--ues-per-cell", type=int, default=2,
                   help="UEs per sector before sites are collapsed (2 × 3 = 6 per site)")
    p.add_argument("--isd", type=float, default=200.0)
    p.add_argument("--slots", type=int, default=1000)
    p.add_argument("--speed", type=float, default=0.8, help="MD speed [m/s]")
    p.add_argument("--tag", default="smoke7")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--bs-rows", type=int, default=2)
    p.add_argument("--bs-cols", type=int, default=4)
    p.add_argument("--ue-ants", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--time-chunk", type=int, default=50,
                   help="Seconds per cluster realization")
    return p.parse_args()


if __name__ == "__main__":
    generate_traces(parse_args())
