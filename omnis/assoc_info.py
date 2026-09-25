"""MD-visible association info: coarse Top-L prune, then joint (model, cell) learning."""

from __future__ import annotations

import math


def init_cell_compute_state(num_cells: int, es_freq: float, user_num: int):
    """Prior before the first slot: empty compute queues, full GPU pool."""
    del user_num  # unused; kept for call-site compatibility
    return {
        int(c): {"q_edge": 0, "n_assoc": 0, "f_total": float(es_freq)}
        for c in range(int(num_cells))
    }


def update_cell_compute_state(cell_dic, gpu_allocation_dic, num_cells: int,
                              es_freq: float, q_edge_by_cell=None):
    """Build the next-slot broadcast from association + compute-queue lengths.

    ``q_edge_by_cell`` maps cell_id -> number of tasks in the ES computing queue.
    GPU share is no longer broadcast: under FIFO edge service the HOL uses the
    full pool when scheduled.
    """
    counts = {int(c): 0 for c in range(int(num_cells))}
    for _user, cell in (cell_dic or {}).items():
        c = int(cell)
        counts[c] = counts.get(c, 0) + 1

    qmap = q_edge_by_cell or {}
    state = {}
    for c in range(int(num_cells)):
        state[c] = {
            "q_edge": int(qmap.get(c, 0)),
            "n_assoc": int(counts.get(c, 0)),
            "f_total": float(es_freq),
        }
    return state


def expected_gpu_if_join(cell_id, cell_compute_state, es_freq: float,
                         user_num: int) -> float:
    """MD-side forecast of GPU frequency when the task reaches the ES.

    Under FIFO the head-of-line task uses the full pool. Contention is the
    broadcast queue length, which enters sojourn as jobs ahead, not as a
    smaller frequency.
    """
    del cell_id, cell_compute_state, user_num
    return float(es_freq)


def compute_queue_broadcast(cell_id, cell_compute_state) -> int:
    """MD-visible computing-queue length at ``cell_id`` [tasks]."""
    c = int(cell_id)
    st = (cell_compute_state or {}).get(c) or {}
    return int(st.get("q_edge", 0))


def _zscore(values):
    """Stabilize coarse mix of radio dB and queue lengths (per-UE normalization)."""
    xs = list(values)
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(len(xs), 1)
    sd = math.sqrt(var) if var > 1e-18 else 1.0
    return [(x - mu) / sd for x in xs]


def coarse_rank_cells(radio_db_by_cell, cell_compute_state, L, es_freq, user_num,
                      w_radio=1.0, w_compute=1.0):
    """Coarse Top-L prune by radio quality and short computing queues.

    Prefer stronger radio and smaller ``q_edge`` (less edge congestion).
    """
    del es_freq, user_num
    cells = [int(c) for c in radio_db_by_cell.keys()]
    if not cells:
        return []
    L = max(1, min(int(L), len(cells)))
    radios = [float(radio_db_by_cell[c]) for c in cells]
    # Negate queue length so larger score = less congested
    qneg = [-float(compute_queue_broadcast(c, cell_compute_state)) for c in cells]
    zr, zq = _zscore(radios), _zscore(qneg)
    wr, wc = float(w_radio), float(w_compute)
    scored = sorted(
        range(len(cells)),
        key=lambda i: wr * zr[i] + wc * zq[i],
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
            "q_edge": compute_queue_broadcast(cid, cell_compute_state),
            "n_assoc": int((cell_compute_state or {}).get(cid, {}).get("n_assoc", 0)),
        })
    return arms
