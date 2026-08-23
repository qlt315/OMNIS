"""MD-visible information for MEC cell association (two-layer design).

Layer 1 — Coarse Top-L prune (complexity reduction, not optimality)
------------------------------------------------------------------
Rank logical cells by MD-obtainable proxies only:

  * radio: local RSRP / SINR-proxy measurement (0 control bytes)
  * compute: ES-broadcast ``n_assoc`` / share hint (see ``comm_model``)

Take the Top-L cells. This is a **coarse** candidate set so the bandit need
not explore every site.

Layer 2 — Joint MAB over ``(model, cell_rank)`` inside Top-L
-----------------------------------------------------------
Model, MCS/link, Acc, edge GPU share, and queues are **coupled** through the
Lyapunov objective. Coarse scores use analytic approximations and last-slot
broadcasts; they cannot certify optimality. The bandit therefore **jointly
learns** model–cell associations inside the pruned set.

MD-invisible (must not drive either layer as god-mode)
-----------------------------------------------------
- Other users' private queues / rewards / arms
- This slot's yet-to-be-computed BCD split
"""

from __future__ import annotations

import math


def init_cell_compute_state(num_cells: int, es_freq: float, user_num: int):
    """Prior before the first allocation: equal share hint per cell."""
    n0 = max(int(user_num), 1)
    share = float(es_freq) / n0
    return {
        int(c): {"n_assoc": 0, "f_share": share, "f_total": float(es_freq)}
        for c in range(int(num_cells))
    }


def update_cell_compute_state(cell_dic, gpu_allocation_dic, num_cells: int,
                              es_freq: float):
    """Build the next-slot broadcast from realized association + GPU alloc."""
    counts = {int(c): 0 for c in range(int(num_cells))}
    for _user, cell in cell_dic.items():
        c = int(cell)
        if c in counts:
            counts[c] += 1
        else:
            counts[c] = counts.get(c, 0) + 1

    state = {}
    for c in range(int(num_cells)):
        n = int(counts.get(c, 0))
        if n <= 0:
            share = float(es_freq)  # idle cell: full pool advertised
        else:
            shares = [
                float(gpu_allocation_dic[u])
                for u, cell in cell_dic.items()
                if int(cell) == c and u in gpu_allocation_dic
            ]
            share = float(sum(shares) / len(shares)) if shares else float(es_freq) / n
        state[c] = {
            "n_assoc": n,
            "f_share": share,
            "f_total": float(es_freq),
        }
    return state


def expected_gpu_if_join(cell_id, cell_compute_state, es_freq: float,
                         user_num: int) -> float:
    """MD-side forecast of GPU frequency if associating to ``cell_id``."""
    c = int(cell_id)
    st = (cell_compute_state or {}).get(c)
    f_tot = float(es_freq)
    if st is None:
        return f_tot / max(int(user_num), 1)
    f_tot = float(st.get("f_total", es_freq))
    n = int(st.get("n_assoc", 0))
    return f_tot / max(n + 1, 1)


def _zscore(values):
    """Stabilize coarse mix of radio dB and log-GPU (per-UE normalization)."""
    xs = list(values)
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(len(xs), 1)
    sd = math.sqrt(var) if var > 1e-18 else 1.0
    return [(x - mu) / sd for x in xs]


def coarse_rank_cells(radio_db_by_cell, cell_compute_state, L, es_freq, user_num,
                      w_radio=1.0, w_compute=1.0):
    """Coarse Top-L prune by radio + broadcast compute (not optimal association).

    Parameters
    ----------
    radio_db_by_cell : dict[int, float]
        Local radio quality (RSRP or SINR-proxy) per logical cell.
    cell_compute_state : dict
        Last ES broadcast (see ``update_cell_compute_state``).
    L : int
        Candidate budget for the joint ``(model, cell_rank)`` bandit.

    Returns
    -------
    list[int]
        Cell ids, best-first. ``cell_rank`` in the MAB indexes this list.
    """
    cells = [int(c) for c in radio_db_by_cell.keys()]
    if not cells:
        return []
    L = max(1, min(int(L), len(cells)))
    radios = [float(radio_db_by_cell[c]) for c in cells]
    gpus = [
        math.log(max(expected_gpu_if_join(
            c, cell_compute_state, es_freq, user_num), 1e-12))
        for c in cells
    ]
    zr, zg = _zscore(radios), _zscore(gpus)
    wr, wc = float(w_radio), float(w_compute)
    scored = sorted(
        range(len(cells)),
        key=lambda i: wr * zr[i] + wc * zg[i],
        reverse=True,
    )
    return [cells[i] for i in scored[:L]]


def md_association_bundle(top_cells, radio_db_by_cell, cell_compute_state,
                          es_freq: float, user_num: int):
    """Package MD-visible observables for arms inside the coarse Top-L."""
    arms = []
    for rank, cell_id in enumerate(top_cells):
        cid = int(cell_id)
        arms.append({
            "cell_rank": int(rank),
            "cell_id": cid,
            "radio_db": float(radio_db_by_cell[cid]),
            "gpu_hat": expected_gpu_if_join(
                cid, cell_compute_state, es_freq, user_num),
            "n_assoc": int((cell_compute_state or {}).get(cid, {}).get("n_assoc", 0)),
        })
    return arms
