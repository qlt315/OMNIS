"""Exogenous CRN streams for matched environments."""
from __future__ import annotations

import hashlib

import numpy as np


def keyed_rng(seed, *parts) -> np.random.RandomState:
    """Deterministic ``RandomState`` for an exogenous draw key."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(seed)).encode("utf-8"))
    for p in parts:
        h.update(b"\0")
        h.update(str(p).encode("utf-8"))
    return np.random.RandomState(int.from_bytes(h.digest(), "little") % (2**32))


def compute_flops_mult(seed, user, t_arrive, stage: str, logstd: float) -> float:
    """Mean-one lognormal multiplier for hardware compute jitter.

    Realized local/edge FLOPs are scaled by ``ξ`` with
    ``log ξ ~ N(-σ²/2, σ²)`` so ``E[ξ]=1``. Captures mild per-inference
    DVFS / cache / runtime variability (not sustained thermal collapse).
    ``logstd=0`` disables (returns 1). Stage is ``\"local\"`` or ``\"edge\"``.
    """
    std = float(logstd)
    if std <= 1e-12:
        return 1.0
    # Mean-preserving: E[exp(Z)]=1 for Z~N(-σ²/2, σ²).
    z = keyed_rng(
        seed, "flops", str(user), float(t_arrive), str(stage)
    ).normal(-0.5 * std * std, std)
    return float(np.exp(z))


def md_pending_wait(pipeline, user: str) -> float:
    """Observed MD-side wait of the HOL pending / active task [s]."""
    at = pipeline.active.get(user)
    if at is not None:
        if at.md_wait_s > 0:
            return float(at.md_wait_s)
        if at.t_arrive is not None:
            return max(0.0, float(pipeline.time_s) - float(at.t_arrive))
        return 0.0
    q = pipeline.pending.get(user)
    if q:
        return max(0.0, float(pipeline.time_s) - float(q[0].t_arrive))
    return 0.0


def exog_seed(agent) -> int:
    return int(getattr(agent, "seed", 0))


def poisson_arrivals(agent, time_slot) -> dict:
    """One Poisson count per user for ``time_slot``, CRN-stable across algos."""
    seed = exog_seed(agent)
    t = int(time_slot)
    out = {}
    for i, user in enumerate(agent.users):
        rate = float(agent.arrival_rate[user])
        out[user] = int(keyed_rng(seed, "arr", t, i).poisson(rate))
    return out


def draw_qos(agent, time_slot, user_idx, arrival_idx) -> dict:
    """One task QoS triple, keyed by (slot, user index, arrival index)."""
    seed = exog_seed(agent)
    rng = keyed_rng(seed, "qos", int(time_slot), int(user_idx), int(arrival_idx))
    d0, d1 = agent.delay_constraint_range
    e0, e1 = agent.energy_constraint_range
    w0, w1 = agent.energy_weight_range
    energy_weight = float(min(max(float(rng.uniform(w0, w1)), 0.0), 1.0))
    return {
        "delay_constraint": float(rng.uniform(d0, d1)),
        "energy_constraint": float(rng.uniform(e0, e1)),
        "energy_weight": energy_weight,
        "delay_weight": 1.0 - energy_weight,
    }


def est_noise_db(seed, time_slot, ue_local, cell_idx, est_err_db) -> float:
    if est_err_db <= 0:
        return 0.0
    return float(est_err_db) * float(
        keyed_rng(seed, "est", int(time_slot), int(ue_local), int(cell_idx)).randn())


def acc_noise(seed, user, t_arrive, model_name, std) -> float:
    if std <= 0:
        return 0.0
    return float(keyed_rng(
        seed, "acc", str(user), float(t_arrive), str(model_name)).normal(0.0, std))
