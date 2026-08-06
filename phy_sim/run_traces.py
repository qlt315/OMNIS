"""Multi-cell SINR trace generation with Sionna's official 3GPP hex-grid.

Uses ``sionna.sys.gen_hexgrid_topology`` (spiral hexagonal grid WITH wraparound,
3 sectors per cell) so interference and topology are 3GPP-compliant and free of
edge effects — replacing the previous hand-rolled 1-ring layout that had no
wraparound and produced unrealistically low, edge-biased SINR.

Key realism choices:
  * Official hex topology + wraparound (no boundary SINR inflation).
  * Frequency reuse factor ``reuse``: only co-channel cells interfere with each
    other. reuse=1 (full reuse) is interference-limited and pins the median SINR
    near -10 dB, which is degenerate for the MCS/accuracy tables. reuse=3
    (textbook sector reuse) lifts the serving-SINR median to ~0 dB, landing the
    links in the discriminative region of the accuracy table.
  * Each SECTOR is a logical serving cell (independent ES / bandwidth pool),
    matching the system model's per-cell ES assumption.
  * Interference is computed per-slot from the actual co-channel UEs (not a
    time-averaged expectation), preserving interference time-variation.
  * Post-MRC scalar SINR: signal and noise combine coherently/incoherently over
    the receive array, so the post-combining noise floor stays a single N0
    (array gain appears only as larger ||h||^2). Documented simplification vs
    per-subcarrier LMMSE post-equalization SINR.

Outputs (per dataset tag):
  sinr_trace_<tag>.csv : slot,cell_id,ue_id,sinr_db   (all candidate links)
  link_info_<tag>.csv  : cell_id,ue_id,distance_m

Smoke test:
    python run_traces.py --cells 7 --ues-per-cell 2 --slots 20 --tag smoke
Full run:
    python run_traces.py --slots 1000 --tag full
"""

import argparse
import csv
import math
import os

import torch
from sionna.phy.channel.tr38901 import PanelArray, UMa, UMi
from sionna.sys import gen_hexgrid_topology

P_TX = 0.1            # transmit power per UE (W), mirrors config.py trans_power
BANDWIDTH = 1e5       # Hz, mirrors config.py total_bandwidth
NOISE_POWER = 10 ** ((-174 + 10 * math.log10(BANDWIDTH)) / 10 - 3)  # W
CARRIER_FREQ = 3.5e9
SLOT_S = 1e-3         # 1 ms slots -> one SINR sample per slot


def make_channel_model(scenario, bs_rows, bs_cols):
    """Uplink SIMO / receive-array channel under 3GPP 38.901.

    Default BS panel is 2x4 = 8 antennas (single-pol). UE stays single-antenna
    omni. The system sim consumes a scalar post-MRC SINR, so spatial
    multiplexing is intentionally out of scope; larger arrays only raise the
    array / diversity gain folded into sinr_db.
    """
    bs_array = PanelArray(num_rows_per_panel=bs_rows, num_cols_per_panel=bs_cols,
                          polarization='single', polarization_type='V',
                          antenna_pattern='38.901',
                          carrier_frequency=CARRIER_FREQ)
    ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                          polarization='single', polarization_type='V',
                          antenna_pattern='omni',
                          carrier_frequency=CARRIER_FREQ)
    cls = {'umi': UMi, 'uma': UMa}[scenario]
    return cls(carrier_frequency=CARRIER_FREQ, o2i_model='low',
               ut_array=ut_array, bs_array=bs_array, direction='uplink',
               enable_pathloss=True, enable_shadow_fading=True)


def generate_traces(args):
    torch.manual_seed(args.seed)

    cm = make_channel_model(args.scenario, args.bs_rows, args.bs_cols)
    n_rx = args.bs_rows * args.bs_cols

    # Official 3GPP hex grid: num_rings=1 -> 7 cells -> 21 sector BSs.
    # Each sector serves num_ut_per_sector UTs. Wraparound is built in.
    ues_per_sector = args.ues_per_cell
    topo = gen_hexgrid_topology(batch_size=1, num_rings=args.num_rings,
                                num_ut_per_sector=ues_per_sector,
                                scenario=args.scenario, isd=args.isd,
                                min_ut_velocity=args.speed,
                                max_ut_velocity=args.speed,
                                downtilt_to_sector_center=True)
    cm.set_topology(*topo)

    h, _ = cm(args.slots, 1.0 / SLOT_S)
    # h: [batch, num_rx=K, num_rx_ant, num_tx=U, num_tx_ant, num_paths, T]
    # Post-MRC uplink gain under spatially-white interference: sum |h|^2 over
    # receive antennas, transmit antennas, and paths (SIMO power combining).
    g_link = h[0].abs().square().sum(dim=(1, 3, 4))      # (K, U, T)
    prx = P_TX * g_link                                   # received power (W)

    K, U = prx.shape[0], prx.shape[1]
    # Sector c serves UTs [c*ups, (c+1)*ups): each sector is a logical cell.
    cell_of = torch.arange(U) // ues_per_sector

    # Frequency reuse: cell c uses frequency band (c % reuse); only cells on the
    # same band interfere. Co-channel set for cell c = {cc != c : cc%reuse == c%reuse}.
    reuse = args.reuse
    cochannel = {c: [cc for cc in range(K)
                     if cc % reuse == c % reuse and cc != c]
                 for c in range(K)}

    sinr_db = torch.empty(K, U, args.slots)
    for c in range(K):
        # Per-slot interference from co-channel UEs in OTHER cells on the same band.
        # Within a cell, UTs are orthogonal (bandwidth slicing); each slice sees the
        # coincident co-channel UTs. We take the actual per-slot sum over co-channel
        # UEs (full co-channel load), which is the standard worst-case reuse model.
        co = cochannel[c]
        if co:
            mask = torch.tensor([cell_of[u].item() in co for u in range(U)],
                                dtype=torch.bool)
            # Average per-slice: divide co-channel power by #UTs per co-channel cell
            interf = prx[c][mask].sum(dim=0) / ues_per_sector
        else:
            interf = torch.zeros(args.slots)
        # Post-MRC scalar SINR: prx already = P*||h||^2 (summed over the array);
        # noise floor stays a single N0 (coherent signal / incoherent-noise MRC).
        sinr_db[c] = 10 * torch.log10(prx[c] / (NOISE_POWER + interf))

    os.makedirs(os.path.dirname(args.out_dir) or '.', exist_ok=True)
    trace_path = os.path.join(args.out_dir, f"sinr_trace_{args.tag}.csv")
    with open(trace_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["slot", "cell_id", "ue_id", "sinr_db"])
        for t in range(args.slots):
            for c in range(K):
                for u in range(U):
                    w.writerow([t, c, u, f"{sinr_db[c, u, t]:.2f}"])

    # Distances use the wraparound-corrected virtual positions for realism.
    bs_loc = topo[1][0]            # (K, 3) serving-sector BS positions
    ut_loc = topo[0][0]            # (U, 3)
    bs_virtual = topo[7][0]        # (K, U, 3) wraparound virtual BS positions
    info_path = os.path.join(args.out_dir, f"link_info_{args.tag}.csv")
    with open(info_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "ue_id", "distance_m"])
        for c in range(K):
            for u in range(U):
                d = (bs_virtual[c, u] - ut_loc[u]).norm().item()
                w.writerow([c, u, f"{d:.1f}"])

    own = sinr_db[cell_of, torch.arange(U)]               # serving-link SINR

    # --- Aggregate the 3 co-sited sectors of each SITE into one logical cell ---
    # The system model assumes `num_cells` independent serving cells (one ES and
    # one bandwidth pool each), while gen_hexgrid_topology yields 3 sector-BSs
    # per hexagonal site. We fold each site's 3 sectors into a single logical
    # cell: that cell's SINR to a UE is the MAX over its 3 sector receivers
    # (standard 3-sector site abstraction), so the output trace has
    # num_sites cells and is directly consumable by the existing system config.
    num_sites = K // 3
    site_sinr = sinr_db.reshape(num_sites, 3, U, args.slots).amax(dim=1)  # (sites,U,T)
    # Re-key UEs to their serving SITE (sector c -> site c//3)
    site_of = cell_of // 3
    own_site = site_sinr[site_of, torch.arange(U)]

    site_trace = os.path.join(args.out_dir, f"sinr_trace_{args.tag}_sites.csv")
    with open(site_trace, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["slot", "cell_id", "ue_id", "sinr_db"])
        for t in range(args.slots):
            for s in range(num_sites):
                for u in range(U):
                    w.writerow([t, s, u, f"{site_sinr[s, u, t]:.2f}"])

    site_info = os.path.join(args.out_dir, f"link_info_{args.tag}_sites.csv")
    with open(site_info, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "ue_id", "distance_m"])
        for s in range(num_sites):
            for u in range(U):
                # distance to the site's (co-sited) sector position
                d = (bs_virtual[s * 3, u] - ut_loc[u]).norm().item()
                w.writerow([s, u, f"{d:.1f}"])

    print(f"scenario={args.scenario} rings={args.num_rings} sectors={K} U={U} "
          f"slots={args.slots} speed={args.speed} m/s reuse={reuse} "
          f"BS-array={n_rx}ant ({args.bs_rows}x{args.bs_cols})")
    print(f"sector-level serving SINR: p5={own.quantile(0.05):.1f} "
          f"median={own.median():.1f} p95={own.quantile(0.95):.1f} dB")
    print(f"site-level ({num_sites} logical cells) serving SINR: "
          f"p5={own_site.quantile(0.05):.1f} median={own_site.median():.1f} "
          f"p95={own_site.quantile(0.95):.1f} dB")
    print(f"wrote {trace_path} and {info_path} (sector-level)")
    print(f"wrote {site_trace} and {site_info} (site-level, use this for system sim)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["umi", "uma"], default="umi")
    p.add_argument("--num-rings", type=int, default=1,
                   help="hex grid rings; 1 -> 7 cells -> 21 sector BSs")
    p.add_argument("--cells", type=int, default=None,
                   help="deprecated alias; use --num-rings (kept for compat)")
    p.add_argument("--ues-per-cell", type=int, default=2,
                   help="UTs per sector (= per logical serving cell)")
    p.add_argument("--isd", type=float, default=200.0, help="inter-site distance (m)")
    p.add_argument("--reuse", type=int, default=3,
                   help="frequency reuse factor; 3 lifts SINR out of the "
                        "interference-limited floor (default 3)")
    p.add_argument("--slots", type=int, default=1000)
    p.add_argument("--speed", type=float, default=0.0, help="UE speed (m/s)")
    p.add_argument("--tag", default="smoke7")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--bs-rows", type=int, default=2, help="BS panel rows")
    p.add_argument("--bs-cols", type=int, default=4, help="BS panel cols -> 8 ant")
    args = p.parse_args()
    return args


if __name__ == "__main__":
    generate_traces(parse_args())
