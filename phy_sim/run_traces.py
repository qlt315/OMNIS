"""Multi-cell SU-MIMO effective-SINR traces (Sionna 3GPP hex-grid).

Uses ``sionna.sys.gen_hexgrid_topology`` (wraparound hex grid, 3 sectors/site).

SU-MIMO uplink (default)
------------------------
  * BS: ``bs_rows × bs_cols`` panel (default 2×4 = 8 Rx).
  * UE: ``ue_ants`` antennas (default 2 Tx) — true SU-MIMO, not SIMO.
  * Per link (sector, UE, slot): sum multipath → H ∈ C^{Nr×Nt}, SVD,
    equal-power on the strongest ``n_layers`` eigenmodes, white N0+I,
    **EESM** of per-layer SINRs → ``sinr_db`` for single-stream MCS/Acc:
        SINR_eff = -β ln( mean_ℓ exp(-SINR_ℓ / β) ),  β=1 by default.
  * Optional column ``se_bps_hz`` = Shannon sum-rate (diagnostics only).

Interference / topology (unchanged spirit)
------------------------------------------
  * Frequency reuse ``reuse``; co-channel sectors only.
  * Intra-sector UEs orthogonal (bandwidth slicing); co-channel power
    averaged per slice.
  * Site-level files: max effective SINR over 3 co-sited sectors.

SIMO fallback: ``--ue-ants 1 --n-layers 1``.

Smoke::
    python run_traces.py --ues-per-cell 2 --slots 20 --tag smoke_mimo
Full (matches Config smoke7 UE pool ≈ 42)::
    python run_traces.py --ues-per-cell 2 --slots 1000 --tag smoke7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import torch
from sionna.phy.channel.tr38901 import PanelArray, UMa, UMi
from sionna.sys import gen_hexgrid_topology

P_TX = 0.1
BANDWIDTH = 1e5
NOISE_POWER = 10 ** ((-174 + 10 * math.log10(BANDWIDTH)) / 10 - 3)
CARRIER_FREQ = 3.5e9
SLOT_S = 1e-3


def make_channel_model(scenario, bs_rows, bs_cols, ue_ants):
    """Uplink SU-MIMO channel under 3GPP 38.901 (Nr × Nt)."""
    bs_array = PanelArray(
        num_rows_per_panel=bs_rows, num_cols_per_panel=bs_cols,
        polarization="single", polarization_type="V",
        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQ)
    ut_array = PanelArray(
        num_rows_per_panel=1, num_cols_per_panel=int(ue_ants),
        polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=CARRIER_FREQ)
    cls = {"umi": UMi, "uma": UMa}[scenario]
    return cls(
        carrier_frequency=CARRIER_FREQ, o2i_model="low",
        ut_array=ut_array, bs_array=bs_array, direction="uplink",
        enable_pathloss=True, enable_shadow_fading=True)


def _cochannel_interference(prx, cell_of, K, U, reuse, ues_per_sector, slots):
    """Per-sector interference power [W]; shape (K, T)."""
    cell_of_list = [int(cell_of[u]) for u in range(U)]
    interf = torch.zeros(K, slots, dtype=prx.dtype, device=prx.device)
    for c in range(K):
        co = {cc for cc in range(K) if cc % reuse == c % reuse and cc != c}
        if not co:
            continue
        mask = torch.tensor(
            [cell_of_list[u] in co for u in range(U)],
            dtype=torch.bool, device=prx.device)
        if mask.any():
            interf[c] = prx[c][mask].sum(dim=0) / float(ues_per_sector)
    return interf


def _svd_eff_sinr(H, interf, n_lay, p_tx, n0, svd_batch, eesm_beta=1.0):
    """H (K,Nr,U,Nt,Tc), interf (K,Tc) → sinr_eff (EESM), se (Shannon sum).

    Per-layer SINR after equal-power SVD; ``sinr_eff`` is Exponential Effective
    SINR Mapping for single-stream MCS/Acc tables (not capacity mapping):
        SINR_eff = -β ln( mean_ℓ exp(-SINR_ℓ / β) ),  β = eesm_beta.
    ``se`` remains Σ log2(1+SINR_ℓ) for diagnostics only.
    """
    K, Nr, U, Nt, Tc = H.shape
    Hb = H.permute(0, 2, 4, 1, 3).contiguous().reshape(K * U * Tc, Nr, Nt)
    sigma2_b = (n0 + interf).clamp_min(1e-30)[:, None, :].expand(
        K, U, Tc).reshape(K * U * Tc)
    p_layer = float(p_tx) / float(n_lay)
    B = Hb.shape[0]
    se = torch.empty(B, dtype=torch.float32, device=Hb.device)
    sinr_eff = torch.empty(B, dtype=torch.float32, device=Hb.device)
    beta = max(float(eesm_beta), 1e-6)
    for s0 in range(0, B, svd_batch):
        s1 = min(s0 + svd_batch, B)
        _u, S, _vh = torch.linalg.svd(Hb[s0:s1], full_matrices=False)
        S_use = S[:, :n_lay].clamp_min(0.0)
        sinr_l = (p_layer * S_use.square()) / sigma2_b[s0:s1, None]
        se[s0:s1] = torch.log2(1.0 + sinr_l).sum(dim=-1)
        # EESM → demapper-equivalent single-stream SNR
        sinr_eff[s0:s1] = -beta * torch.log(
            torch.exp(-sinr_l / beta).mean(dim=-1).clamp_min(1e-30))
    return sinr_eff.reshape(K, U, Tc), se.reshape(K, U, Tc)


def generate_traces(args):
    torch.manual_seed(args.seed)
    ue_ants = max(1, int(args.ue_ants))
    n_layers_req = max(1, int(args.n_layers))
    time_chunk = max(1, int(args.time_chunk))
    svd_batch = max(256, int(args.svd_batch))

    cm = make_channel_model(
        args.scenario, args.bs_rows, args.bs_cols, ue_ants)
    n_rx = args.bs_rows * args.bs_cols
    ues_per_sector = args.ues_per_cell
    topo = gen_hexgrid_topology(
        batch_size=1, num_rings=args.num_rings,
        num_ut_per_sector=ues_per_sector,
        scenario=args.scenario, isd=args.isd,
        min_ut_velocity=args.speed, max_ut_velocity=args.speed,
        downtilt_to_sector_center=True)
    cm.set_topology(*topo)

    print(
        f"generating CIR in chunks of {time_chunk} slots "
        f"(SU-MIMO {n_rx}x{ue_ants}, L≤{n_layers_req}) …",
        flush=True)

    # Probe one slot for shapes (then discard)
    h0, _ = cm(1, 1.0 / SLOT_S)
    K, Nr, U, Nt, _Np, _ = h0[0].shape
    del h0
    if Nt != ue_ants:
        raise RuntimeError(f"expected Nt={ue_ants}, got Nt={Nt}")
    n_lay = max(1, min(n_layers_req, Nr, Nt))
    reuse = int(args.reuse)
    cell_of = torch.arange(U) // ues_per_sector
    T = int(args.slots)

    sinr_db = torch.empty(K, U, T, dtype=torch.float32)
    se_bps = torch.empty(K, U, T, dtype=torch.float32)

    for t0 in range(0, T, time_chunk):
        t1 = min(t0 + time_chunk, T)
        tc = t1 - t0
        h, _ = cm(tc, 1.0 / SLOT_S)
        H = h[0].sum(dim=4).to(torch.complex64).cpu()
        del h
        g_fro = H.abs().square().sum(dim=(1, 3))
        prx = P_TX * g_fro
        interf = _cochannel_interference(
            prx, cell_of, K, U, reuse, ues_per_sector, tc)
        sinr_lin, se = _svd_eff_sinr(
            H, interf, n_lay, P_TX, NOISE_POWER, svd_batch)
        sinr_db[:, :, t0:t1] = 10.0 * torch.log10(sinr_lin.clamp_min(1e-30))
        se_bps[:, :, t0:t1] = se
        del H, g_fro, prx, interf, sinr_lin, se
        if (t0 // time_chunk) % 5 == 0:
            print(f"  slots {t0}:{t1}/{T}", flush=True)

    # shapes for meta / prints (reuse from loop)
    Nt = ue_ants

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "phy": "su_mimo",
        "direction": "uplink",
        "scenario": args.scenario,
        "bs_rows": args.bs_rows,
        "bs_cols": args.bs_cols,
        "n_rx": n_rx,
        "ue_ants": ue_ants,
        "n_tx": int(Nt),
        "n_layers": n_lay,
        "effective_sinr": "eesm",
        "eesm_beta": 1.0,
        "formula": "SINR_eff = -beta * ln(mean_l exp(-SINR_l/beta))",
        "p_tx_W": P_TX,
        "bandwidth_Hz": BANDWIDTH,
        "noise_W": NOISE_POWER,
        "reuse": reuse,
        "num_rings": args.num_rings,
        "ues_per_sector": ues_per_sector,
        "slots": args.slots,
        "speed_mps": args.speed,
        "seed": args.seed,
        "tag": args.tag,
        "time_chunk": time_chunk,
        "note": (
            "sinr_db is EESM of per-layer post-SVD SINRs for single-stream "
            "MCS/Acc tables; se_bps_hz is Shannon sum SE (diagnostics). "
            "CIR is generated in time_chunk windows to bound RAM; with "
            "speed>0, fading is not continuous across chunk boundaries."
        ),
    }
    meta_path = os.path.join(args.out_dir, f"sinr_trace_{args.tag}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    def write_sinr_csv(path, cube_db, cube_se, n_cells):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "slot", "cell_id", "ue_id", "sinr_db", "se_bps_hz", "n_layers",
            ])
            for t in range(T):
                for c in range(n_cells):
                    for u in range(U):
                        w.writerow([
                            t, c, u,
                            f"{float(cube_db[c, u, t]):.2f}",
                            f"{float(cube_se[c, u, t]):.4f}",
                            n_lay,
                        ])

    trace_path = os.path.join(args.out_dir, f"sinr_trace_{args.tag}.csv")
    write_sinr_csv(trace_path, sinr_db, se_bps, K)

    bs_virtual = topo[7][0]
    ut_loc = topo[0][0]
    info_path = os.path.join(args.out_dir, f"link_info_{args.tag}.csv")
    with open(info_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "ue_id", "distance_m"])
        for c in range(K):
            for u in range(U):
                d = (bs_virtual[c, u] - ut_loc[u]).norm().item()
                w.writerow([c, u, f"{d:.1f}"])

    own = sinr_db[cell_of, torch.arange(U)]
    num_sites = K // 3
    site_sinr = sinr_db.reshape(num_sites, 3, U, T).amax(dim=1)
    site_se = se_bps.reshape(num_sites, 3, U, T).amax(dim=1)
    site_of = cell_of // 3
    own_site = site_sinr[site_of, torch.arange(U)]

    site_trace = os.path.join(args.out_dir, f"sinr_trace_{args.tag}_sites.csv")
    write_sinr_csv(site_trace, site_sinr, site_se, num_sites)
    site_info = os.path.join(args.out_dir, f"link_info_{args.tag}_sites.csv")
    with open(site_info, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "ue_id", "distance_m"])
        for s in range(num_sites):
            for u in range(U):
                d = (bs_virtual[s * 3, u] - ut_loc[u]).norm().item()
                w.writerow([s, u, f"{d:.1f}"])

    print(
        f"phy=SU-MIMO {n_rx}x{Nt} layers={n_lay} scenario={args.scenario} "
        f"rings={args.num_rings} sectors={K} sites={num_sites} U={U} "
        f"slots={T} speed={args.speed} reuse={reuse}",
        flush=True)
    print(
        f"sector serving SINR_eff: p5={own.quantile(0.05):.1f} "
        f"median={own.median():.1f} p95={own.quantile(0.95):.1f} dB",
        flush=True)
    print(
        f"site serving SINR_eff: p5={own_site.quantile(0.05):.1f} "
        f"median={own_site.median():.1f} p95={own_site.quantile(0.95):.1f} dB",
        flush=True)
    print(f"wrote {trace_path} / {info_path}", flush=True)
    print(f"wrote {site_trace} / {site_info} (system sim uses *_sites)", flush=True)
    print(f"wrote {meta_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["umi", "uma"], default="umi")
    p.add_argument("--num-rings", type=int, default=1)
    p.add_argument("--cells", type=int, default=None)
    p.add_argument("--ues-per-cell", type=int, default=2)
    p.add_argument("--isd", type=float, default=200.0)
    p.add_argument("--reuse", type=int, default=3)
    p.add_argument("--slots", type=int, default=1000)
    p.add_argument("--speed", type=float, default=0.0)
    p.add_argument("--tag", default="smoke7")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--bs-rows", type=int, default=2)
    p.add_argument("--bs-cols", type=int, default=4)
    p.add_argument("--ue-ants", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--time-chunk", type=int, default=20,
                   help="CIR+SVD slots per window (lower → less RAM)")
    p.add_argument("--svd-batch", type=int, default=2048,
                   help="matrices per torch.linalg.svd call")
    return p.parse_args()


if __name__ == "__main__":
    generate_traces(parse_args())
