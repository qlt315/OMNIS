"""Common helpers to retrofit discrete task queues onto baseline agents."""

from __future__ import annotations

import numpy as np

from omnis.radio_obs import payload_bits
from omnis.task_pipeline import TaskPipeline


def init_task_pipeline(agent, config):
    agent.dpp_task_scale = float(getattr(config, "dpp_task_scale", 3.0))
    agent.pipeline = TaskPipeline(agent.users, agent.num_cells)
    agent.backlog = {u: 0.0 for u in agent.users}
    agent.energy_queue = agent.pipeline.energy_queue
    if not hasattr(agent, "_last_bandwidth"):
        agent._last_bandwidth = {}
    if not hasattr(agent, "_last_gpu"):
        agent._last_gpu = {}
    if not hasattr(agent, "learn_delay_slots"):
        agent.learn_delay_slots = int(getattr(config, "learn_delay_slots", 0) or 0)
    if not hasattr(agent, "_pending_learn"):
        agent._pending_learn = None
    if not hasattr(agent, "acc_noise_std"):
        agent.acc_noise_std = float(getattr(config, "acc_noise_std", 0.0))


def sticky_or_select(agent, user, cand_cells, pick_fn):
    """If MD has an unfinished task, keep locked (model, cell); else pick_fn()."""
    locked_cell = agent.pipeline.locked_cell(user)
    locked_model = agent.pipeline.locked_model(user)
    if locked_cell is not None and locked_model is not None:
        if locked_cell in cand_cells:
            cell_rank = int(cand_cells.index(locked_cell))
        else:
            cell_rank = 0
        return locked_model, locked_cell, cell_rank, True
    model_name, cell_id, cell_rank = pick_fn()
    return model_name, cell_id, cell_rank, False


def local_queue_len(agent, user):
    return float(agent.pipeline.tx_queue_len(user))


def composite_backlog(agent, user, cell_id=None):
    return float(agent.pipeline.composite_backlog(user, cell_id=cell_id))


def get_average_and_std_metrics(agent):
    """Slot averages for queues; completion-conditioned averages for task QoS.

    Accuracy/delay/energy/reward/violations are averaged only over (user,slot)
    samples with ``served==1``. Backlog/arrivals/served stay over all slots.
    """
    task_keys = ["delay", "energy", "accuracy", "reward", "is_vio", "vio_degree"]
    queue_keys = ["backlog", "energy_queue", "arrivals", "served"]
    task_sums = {m: 0.0 for m in task_keys}
    task_vals = {m: [] for m in task_keys}
    queue_sums = {m: 0.0 for m in queue_keys}
    queue_vals = {m: [] for m in queue_keys}
    n_complete = 0
    n_slot = 0

    for user in agent.users:
        for t in range(agent.time_slot_num):
            n_slot += 1
            served = float(agent.instant_metrics[user]["served"][t])
            for m in queue_keys:
                v = float(agent.instant_metrics[user][m][t])
                queue_sums[m] += v
                queue_vals[m].append(v)
            if served >= 0.5:
                n_complete += 1
                for m in task_keys:
                    v = float(agent.instant_metrics[user][m][t])
                    task_sums[m] += v
                    task_vals[m].append(v)

    def _mean(s, n):
        return float(s / n) if n > 0 else 0.0

    def _std(vals):
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    agent.average_metrics = {
        "latency": _mean(task_sums["delay"], n_complete),
        "energy": _mean(task_sums["energy"], n_complete),
        "accuracy": _mean(task_sums["accuracy"], n_complete),
        "reward": _mean(task_sums["reward"], n_complete),
        "vio_prob": _mean(task_sums["is_vio"], n_complete),
        "vio_sum": _mean(task_sums["vio_degree"], n_complete),
        "backlog_bits": _mean(queue_sums["backlog"], n_slot),
        "energy_queue": _mean(queue_sums["energy_queue"], n_slot),
        "arrival_bits": _mean(queue_sums["arrivals"], n_slot),
        "served_bits": _mean(queue_sums["served"], n_slot),
    }
    agent.std_metrics = {
        "latency": _std(task_vals["delay"]),
        "energy": _std(task_vals["energy"]),
        "accuracy": _std(task_vals["accuracy"]),
        "reward": _std(task_vals["reward"]),
        "vio_prob": _std(task_vals["is_vio"]),
        "vio_sum": _std(task_vals["vio_degree"]),
        "backlog_bits": _std(queue_vals["backlog"]),
        "energy_queue": _std(queue_vals["energy_queue"]),
    }
    agent.action_freq = agent.action_freq / max(agent.time_slot_num, 1)


def task_dpp_drift(agent, user, model_name, mcs_idx, energy_hat, snr_db=0.0,
                   cell_id=None):
    """Lyapunov drift: same uplink forecast and grant wait as admission."""
    from omnis.radio_obs import radio_grant_wait
    from omnis.task_pipeline import STAGE_EDGE

    pipe = getattr(agent, "pipeline", None)
    if cell_id is None and pipe is not None:
        locked = pipe.locked_cell(user)
        if locked is not None:
            cell_id = int(locked)

    bw_fn = getattr(agent, "_bw_override", None)
    if bw_fn is not None and cell_id is not None:
        bandwidth_hat = float(bw_fn(user, cell_id))
    elif hasattr(agent, "_forecast_uplink_bw") and cell_id is not None:
        last = agent._last_bandwidth.get(
            user, agent.total_bandwidth / max(agent.user_num, 1))
        bandwidth_hat = float(agent._forecast_uplink_bw(user, cell_id, last))
    else:
        bandwidth_hat = agent._last_bandwidth.get(
            user, agent.total_bandwidth / agent.user_num)
    se_eff = max(agent.mcs_table.goodput_se(model_name, mcs_idx, snr_db), 1e-12)
    es = agent.es_params
    from omnis.compute_stats import ensure_compute_stats
    stats = ensure_compute_stats(agent)
    local_d, _local_e = stats.local_hat(user, model_name)
    bits = agent.pipeline.uplink_residual_bits(user)
    if bits <= 1e-9:
        bits = agent._payload_bits(model_name)
    tx_d = bits / max(bandwidth_hat * se_eff, 1e-12)
    # Concurrent GPU share from MD-visible broadcast / last grant — never
    # assume the full pool F^e (that would leak an oracle ES view).
    from omnis.assoc_info import expected_gpu_if_join
    from omnis.gpu_alloc import forecast_gpu_share
    at = pipe.active.get(user) if pipe is not None else None
    in_edge = at is not None and at.stage == STAGE_EDGE
    if cell_id is not None and hasattr(agent, "_cell_compute_state"):
        if in_edge:
            q = int((agent._cell_compute_state or {}).get(
                int(cell_id), {}).get("q_edge", 0))
            gpu_hat = forecast_gpu_share(
                q, float(es["freq"]), already_in_edge=True)
        else:
            gpu_hat = float(expected_gpu_if_join(
                cell_id, agent._cell_compute_state,
                es["freq"], getattr(agent, "user_num", 1)))
    else:
        gpu_hat = float(agent._last_gpu.get(
            user, float(es["freq"]) / max(getattr(agent, "user_num", 1), 1)))
    gpu_hat = max(gpu_hat, 1e-12)
    edge_d, _edge_e = stats.edge_hat(model_name, gpu_hat)
    tau_s = max(
        local_d + tx_d + edge_d
        + radio_grant_wait(local_d, getattr(agent, "slot_duration", 1.0)),
        1e-6)
    scale = float(getattr(agent, "dpp_task_scale", 3.0))
    service_n = (agent.slot_duration / tau_s) / scale
    arrivals_n = float(agent.arrival_rate[user]) / scale
    q_n = float(np.tanh(
        agent.pipeline.composite_backlog(user, cell_id=cell_id) / scale))
    return q_n * (service_n - arrivals_n)
