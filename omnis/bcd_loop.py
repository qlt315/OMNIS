"""Shared BCD slot loop and per-cell parallel helpers.

After user–cell association, cells are independent: each cell has its own
bandwidth and GPU pools, and MCS selection only depends on that cell's users.
``allocate_bandwidth_all_cells``, ``gpu_resource_allocation_all_cells``, and
``mcs_selection_all_cells`` therefore parallelize one task per cell via
``ThreadPoolExecutor`` (workers capped sensibly).

BCD optimizations (ε / max_iter unchanged; set on the agent / Config):
  1. Warm-start MCS from ``agent._last_mcs`` (else random on the first slot).
  2. GPU allocation once per slot (independent of MCS / BW); reused every iter.
  3. Per-cell parallelism as above.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from scipy.special import erf

# Cap workers: avoid oversubscription on small machines / many cells.
_MAX_BCD_WORKERS = 8


def bcd_max_workers(n_cells: int) -> int:
    """Sensible ThreadPoolExecutor size for per-cell BCD subproblems."""
    cpu = os.cpu_count() or 4
    return max(1, min(int(n_cells), cpu, _MAX_BCD_WORKERS))


def _merge_cell_dicts(cell_groups, work_fn):
    """Run ``work_fn(users)`` per cell (parallel when >1 cell) and merge dicts.

    Cells are independent after association; see module docstring.
    """
    items = [(cell, users) for cell, users in cell_groups.items() if users]
    if not items:
        return {}
    if len(items) == 1:
        return dict(work_fn(items[0][1]))

    out = {}
    workers = bcd_max_workers(len(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work_fn, users) for _cell, users in items]
        for fut in futures:
            out.update(fut.result())
    return out


def allocate_bandwidth_all_cells(agent, task_dic, model_selection_dic, trans_rate_dic,
                                 phy_choice_dic, cell_dic, snr_dic=None):
    """Per-cell bandwidth allocation (parallel over cells)."""
    groups = agent._users_by_cell(cell_dic)
    return _merge_cell_dicts(
        groups,
        lambda users: agent.allocate_bandwidth(
            task_dic, model_selection_dic, trans_rate_dic, phy_choice_dic,
            users=users, snr_dic=snr_dic),
    )


def gpu_resource_allocation_all_cells(agent, task_dic, model_selection_dic, cell_dic):
    """Per-cell GPU allocation (parallel over cells)."""
    groups = agent._users_by_cell(cell_dic)
    return _merge_cell_dicts(
        groups,
        lambda users: agent.gpu_resource_allocation(
            task_dic, model_selection_dic, users=users),
    )


def mcs_selection_all_cells(agent, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
                            local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic,
                            cell_dic):
    """Per-cell MCS selection (parallel over cells)."""
    groups = agent._users_by_cell(cell_dic)
    return _merge_cell_dicts(
        groups,
        lambda users: agent.mcs_selection(
            task_dic, snr_dic, trans_rate_dic, model_selection_dic,
            local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic,
            users=users),
    )


def _warm_start_mcs(agent):
    """Previous-slot MCS when available; otherwise random (first slot)."""
    last = getattr(agent, "_last_mcs", None) or {}
    avail = set(agent.available_mcs)
    init = {}
    for user in agent.users:
        m = last.get(user)
        if m in avail:
            init[user] = m
        else:
            init[user] = random.choice(agent.available_mcs)
    return init


def _track_bcd_iters(agent, bcd_iter: int) -> None:
    """Running average of BCD iterations per slot on ``agent.bcd_iters``."""
    total = float(getattr(agent, "_bcd_iters_total", 0.0)) + float(bcd_iter)
    n = int(getattr(agent, "_bcd_n_slots", 0)) + 1
    agent._bcd_iters_total = total
    agent._bcd_n_slots = n
    agent.bcd_iters = total / n


def run_bcd_slot(agent, task_dic, model_selection_dic, trans_rate_dic,
                 local_overhead_dic, snr_dic, cell_dic):
    """Run one slot of BCD: warm-start MCS, GPU once, iterate BW ↔ MCS.

    Updates ``agent.bcd_time``, ``agent.bcd_iters``, and ``agent._last_mcs``.

    Returns
    -------
    dict with keys bandwidth, gpu, phy_choice, total_overhead, bcd_iter, bcd_obj
    """
    # perf_counter: monotonic; still wall-ish under contention, but better than
    # time.time(). Cross-algo BCD bars are normalized at plot time (shared median).
    t_bcd = time.perf_counter()
    bcd_obj_last = float("inf")
    bcd_iter = 1
    phy_choice_dic = {}
    bandwidth_allocation_dic = {}
    total_overhead_dic = {}
    bcd_obj = float("inf")

    # GPU does not depend on MCS / BW — solve once per slot, reuse every iter.
    gpu_allocation_dic = gpu_resource_allocation_all_cells(
        agent, task_dic, model_selection_dic, cell_dic)

    while True:
        if bcd_iter == 1:
            init_mcs_dic = _warm_start_mcs(agent)
            bandwidth_allocation_dic = allocate_bandwidth_all_cells(
                agent, task_dic, model_selection_dic, trans_rate_dic,
                init_mcs_dic, cell_dic, snr_dic=snr_dic)
        else:
            bandwidth_allocation_dic = allocate_bandwidth_all_cells(
                agent, task_dic, model_selection_dic, trans_rate_dic,
                phy_choice_dic, cell_dic, snr_dic=snr_dic)

        phy_choice_dic = mcs_selection_all_cells(
            agent, task_dic, snr_dic, trans_rate_dic, model_selection_dic,
            local_overhead_dic, bandwidth_allocation_dic, gpu_allocation_dic,
            cell_dic)

        trans_overhead_dic = agent.get_trans_overhead(
            trans_rate_dic, model_selection_dic, bandwidth_allocation_dic,
            phy_choice_dic, snr_dic=snr_dic)
        edge_overhead_dic = agent.get_edge_overhead(
            model_selection_dic, gpu_allocation_dic)
        queue_wait_dic = {
            user: agent.backlog[user] / (
                bandwidth_allocation_dic[user] * agent._goodput_se(
                    user, model_selection_dic[user]["model"],
                    phy_choice_dic[user], snr_dic))
            for user in agent.users
        }
        total_overhead_dic = agent.get_total_overhead(
            local_overhead_dic, trans_overhead_dic, edge_overhead_dic,
            queue_wait_dic)

        # BCD objective: QoS penalties only (no Acc-table term; Acc is env-only)
        bcd_delay_penalty = sum(
            erf(total_overhead_dic[user]["delay"] - task_dic[user]["delay_constraint"])
            for user in agent.users)
        bcd_energy_penalty = sum(
            erf(total_overhead_dic[user]["energy"] - task_dic[user]["energy_constraint"])
            for user in agent.users)
        bcd_obj = bcd_delay_penalty + bcd_energy_penalty

        if abs(bcd_obj - bcd_obj_last) <= agent.bcd_flag or bcd_iter >= agent.bcd_max_iter:
            break
        bcd_iter += 1
        bcd_obj_last = bcd_obj

    agent.bcd_time += time.perf_counter() - t_bcd
    agent._last_mcs = dict(phy_choice_dic)
    _track_bcd_iters(agent, bcd_iter)

    return {
        "bandwidth": bandwidth_allocation_dic,
        "gpu": gpu_allocation_dic,
        "phy_choice": phy_choice_dic,
        "total_overhead": total_overhead_dic,
        "bcd_iter": bcd_iter,
        "bcd_obj": bcd_obj,
    }
