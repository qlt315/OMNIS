"""Conference-style continuous GPU frequency allocation among edge-stage tasks."""

from __future__ import annotations

from typing import Dict


def allocate_edge_gpu(
    pipeline,
    es_params: dict,
    task_dic: dict,
    model_selection_dic: dict,
    users,
    num_cells: int,
    compute_stats,
) -> Dict[str, float]:
    """Closed-form share of the cell GPU pool among concurrent edge tasks.

    Minimizes ``sum_m (A_m / f_m + B_m f_m^2)`` subject to ``sum f = F`` by
    taking the unconstrained stationary point ``f_m* = (A_m / (2 B_m))^{1/3}``
    (conference P2.2') and normalizing onto the pool ``F^e``.

    A/B use only online ``W_hat`` / ``E_norm_hat`` from ``compute_stats``
    (no FLOPs / κ / power-coefficient formulas).
    """
    users = list(users)
    out = {u: 0.0 for u in users}
    if pipeline is None:
        n = max(len(users), 1)
        f_tot = float(es_params["freq"])
        return {u: f_tot / n for u in users}

    if compute_stats is None:
        raise ValueError(
            "allocate_edge_gpu requires compute_stats; FLOPs oracle path removed")

    f_tot = float(es_params["freq"])

    for cell in range(int(num_cells)):
        edge_users = []
        seen = set()
        for task in pipeline.edge_q[cell]:
            u = task.user
            if u in seen:
                continue
            seen.add(u)
            edge_users.append(u)
        if not edge_users:
            continue

        weights = {}
        n_edge = len(edge_users)
        f_eq = f_tot / max(n_edge, 1)
        for u in edge_users:
            model = None
            if u in model_selection_dic:
                model = model_selection_dic[u].get("model")
            at = pipeline.active.get(u)
            if model is None and at is not None:
                model = at.model_name
            td = task_dic.get(u) or {}
            if at is not None:
                omega_d = float(td.get("delay_weight", at.delay_weight))
                omega_e = float(td.get("energy_weight", at.energy_weight))
            else:
                omega_d = float(td.get("delay_weight", 0.5))
                omega_e = float(td.get("energy_weight", 0.5))

            if model is None:
                weights[u] = 1.0
                continue

            W_full, E_norm = compute_stats.edge_work(model)
            # Mid-edge: remaining work ≈ residual_wall * equal-share f
            # (ES-visible residual; no FLOPs). Fresh admits use full W_hat.
            from omnis.task_pipeline import STAGE_EDGE
            if (at is not None and at.stage == STAGE_EDGE
                    and at.t_edge_start is not None
                    and float(at.residual) > 1e-12):
                W = max(float(at.residual) * f_eq, 1e-12)
                frac = min(W / max(W_full, 1e-12), 1.0)
                E_norm = max(E_norm * frac, 1e-18)
            else:
                W = max(W_full, 1e-12)

            A = omega_d * W
            B = omega_e * max(E_norm, 1e-18)

            if B <= 1e-18:
                weights[u] = max(A, 1e-12)
            else:
                weights[u] = (A / (2.0 * B)) ** (1.0 / 3.0)

        s = sum(weights.values())
        if s <= 1e-18:
            share = f_tot / len(edge_users)
            for u in edge_users:
                out[u] = share
        else:
            for u in edge_users:
                out[u] = f_tot * weights[u] / s
    return out


def forecast_gpu_share(
    q_edge: int,
    es_freq: float,
    already_in_edge: bool = False,
) -> float:
    """MD/ES forecast of concurrent GPU share given broadcast queue length."""
    q = max(int(q_edge), 0)
    if already_in_edge:
        n = max(q, 1)
    else:
        n = max(q + 1, 1)
    return float(es_freq) / float(n)
