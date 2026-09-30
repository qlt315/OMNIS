"""Admission-time proxy vs realized delay/energy components for calibration."""

from __future__ import annotations

from typing import Dict

from omnis.exog import md_pending_wait
from omnis.compute_stats import (
    ensure_compute_stats, local_edge_compute_parts, tx_power_forecast)


def estimate_components(agent, user: str, model_name: str, cell_id: int,
                        task_u: dict, snr_db: float) -> Dict[str, float]:
    """MD decision-time hats (local/edge EWMA + wireless uplink formula)."""
    stats = ensure_compute_stats(agent)
    gpu_hat = float(agent._gpu_hat_for_association(user, cell_id=cell_id))
    local_d, local_e, edge_d, edge_e = local_edge_compute_parts(
        agent, user, model_name, gpu_hat)

    mcs = agent.forward_sim_mcs(user, snr_db, model_name, task_u, cell_id=cell_id)
    bw = float(agent._forecast_uplink_bw(
        user, cell_id,
        agent._last_bandwidth.get(
            user, agent.total_bandwidth / max(agent.user_num, 1))))
    se = max(agent.mcs_table.delay_se(model_name, mcs, snr_db), 1e-12)
    bits = float(agent._payload_bits(model_name))
    tx_payload_d = bits / max(bw * se, 1e-12)
    grant = float(agent._radio_grant_wait(local_d))
    p_tx = float(tx_power_forecast(agent, user, cell_id=cell_id, bw_hz=bw))
    tx_e = p_tx * tx_payload_d
    queue_d = float(md_pending_wait(agent.pipeline, user))

    return {
        "mcs": float(mcs),
        "snr_db": float(snr_db),
        "gpu_hat": float(gpu_hat),
        "bw_hat": float(bw),
        "local_d": float(local_d),
        "local_e": float(local_e),
        "tx_d": float(tx_payload_d + grant),
        "tx_e": float(tx_e),
        "tx_payload_d": float(tx_payload_d),
        "grant_d": float(grant),
        "edge_d": float(edge_d),
        "edge_e": float(edge_e),
        "queue_d": float(queue_d),
        "queue_e": 0.0,
        "service_d": float(local_d + grant + tx_payload_d + edge_d),
        "sojourn_d": float(local_d + grant + tx_payload_d + edge_d + queue_d),
        "total_e": float(local_e + tx_e + edge_e),
        "n_local": float(stats._local_n.get((user, model_name), 0)),
        "n_edge": float(stats._edge_n.get(model_name, 0)),
    }


def _recompute_aggregates(hats: dict) -> None:
    hats["tx_d"] = float(hats["tx_payload_d"] + hats["grant_d"])
    hats["service_d"] = float(
        hats["local_d"] + hats["grant_d"] + hats["tx_payload_d"] + hats["edge_d"])
    hats["sojourn_d"] = float(hats["service_d"] + hats["queue_d"])
    hats["total_e"] = float(hats["local_e"] + hats["tx_e"] + hats["edge_e"])


def attach_admit_proxy(agent, task, task_u: dict, snr_db: float) -> None:
    """Store MD admission hats on ``task.meta['proxy']``."""
    if task is None or task.model_name is None or task.cell_id is None:
        return
    hats = estimate_components(
        agent, task.user, task.model_name, int(task.cell_id), task_u, snr_db)
    hats["queue_d"] = float(task.md_wait_s)
    hats["md_wait_at_admit"] = float(task.md_wait_s)
    _recompute_aggregates(hats)
    hats["tx_payload_d_md"] = float(hats["tx_payload_d"])
    hats["tx_e_md"] = float(hats["tx_e"])
    hats["bw_hat_md"] = float(hats["bw_hat"])
    hats["service_d_md"] = float(hats["service_d"])
    hats["sojourn_d_md"] = float(hats["sojourn_d"])
    hats["total_e_md"] = float(hats["total_e"])
    task.meta["proxy"] = hats


def realized_components(task) -> Dict[str, float]:
    queue_d = float(task.md_wait_s) + float(task.edge_wait_s)
    return {
        "local_d": float(task.local_s),
        "local_e": float(task.local_e),
        "tx_d": float(task.tx_s),
        "tx_e": float(task.tx_e),
        "edge_d": float(task.edge_s),
        "edge_e": float(task.edge_e),
        "queue_d": queue_d,
        "queue_e": 0.0,
        "md_wait_d": float(task.md_wait_s),
        "edge_wait_d": float(task.edge_wait_s),
        "service_d": float(task.service_delay),
        "e2e_d": float(task.e2e_delay),
        "total_e": float(task.total_energy),
        "edge_f_bar": float(getattr(task, "edge_f_bar", 0.0)),
    }


def record_completion(agent, task) -> None:
    """Append one admit-vs-realized row when a task completes.

    ES uplink is measured (timestamps / grant accounting), so ES composite
    hats plug in realized airtime rather than an uplink forecast.
    """
    hats = (task.meta or {}).get("proxy")
    if not hats:
        return
    real = realized_components(task)
    t_done = float(task.t_done) if task.t_done is not None else float("nan")
    t_admit = float(task.t_admit) if task.t_admit is not None else float("nan")
    tx_md = float(hats["tx_payload_d"])
    tx_e_md = float(hats["tx_e"])
    grant = float(hats.get("grant_d", 0.0))
    service_es = float(
        hats["local_d"] + grant + real["tx_d"] + hats["edge_d"])
    sojourn_es = float(service_es + hats["queue_d"])
    total_e_es = float(hats["local_e"] + real["tx_e"] + hats["edge_e"])
    row = {
        "user": task.user,
        "model": task.model_name,
        "cell_id": int(task.cell_id) if task.cell_id is not None else -1,
        "t_arrive": float(task.t_arrive),
        "t_admit": t_admit,
        "t_done": t_done,
        "slot": int(np_floor_slot(t_done)),
        "mcs_hat": float(hats.get("mcs", 0.0)),
        "snr_db_hat": float(hats.get("snr_db", 0.0)),
        "gpu_hat": float(hats.get("gpu_hat", 0.0)),
        "bw_hat": float(hats.get("bw_hat", 0.0)),
        "hat_local_d": hats["local_d"],
        "real_local_d": real["local_d"],
        "hat_tx_d": tx_md,
        "real_tx_d": real["tx_d"],
        "hat_grant_d": grant,
        "real_radio_wait_d": float(getattr(task, "radio_wait_s", 0.0)),
        "hat_edge_d": hats["edge_d"],
        "real_edge_d": real["edge_d"],
        "hat_queue_d": hats["queue_d"],
        "real_queue_d": real["queue_d"],
        "hat_service_d": float(hats["service_d"]),
        "hat_service_d_es": service_es,
        "real_service_d": real["service_d"],
        "hat_sojourn_d": float(hats["sojourn_d"]),
        "hat_sojourn_d_es": sojourn_es,
        "real_e2e_d": real["e2e_d"],
        "hat_local_e": hats["local_e"],
        "real_local_e": real["local_e"],
        "hat_tx_e": tx_e_md,
        "real_tx_e": real["tx_e"],
        "hat_edge_e": hats["edge_e"],
        "real_edge_e": real["edge_e"],
        "hat_queue_e": hats["queue_e"],
        "real_queue_e": real["queue_e"],
        "hat_total_e": float(hats["total_e"]),
        "hat_total_e_es": total_e_es,
        "real_total_e": real["total_e"],
        "md_wait_at_admit": float(hats.get("md_wait_at_admit", 0.0)),
        "real_md_wait_d": real["md_wait_d"],
        "real_edge_wait_d": real["edge_wait_d"],
        "edge_f_bar": real.get("edge_f_bar", 0.0),
        "n_local": float(hats.get("n_local", 0.0)),
        "n_edge": float(hats.get("n_edge", 0.0)),
    }
    if not hasattr(agent, "proxy_calib"):
        agent.proxy_calib = []
    agent.proxy_calib.append(row)


def np_floor_slot(t: float) -> int:
    if t != t:
        return -1
    return max(0, int(t))
