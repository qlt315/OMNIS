"""Control-plane bytes / RTT → equivalent comm time (not in local decision_time)."""

from __future__ import annotations

# --- message payloads [bytes] ---
B_ACTION = 4          # model_idx + cell_rank
B_REWARD = 8          # scalar learning feedback (piggyback next uplink)
B_ALLOC = 16          # bandwidth + GPU (+ MCS ack)
B_CONTEXT = 32        # QoS weights / constraints / coarse SNR
B_FLOAT = 4
# MD→ES: branch + measured local delay/energy (offload report)
B_LOCAL_MEAS = 12
# ES→MD: concurrency + f_bar + edge delay/energy (completion report)
B_EDGE_MEAS = 20
# Per logical cell: n_assoc + f_share (MD-visible compute for association)
B_CELL_COMPUTE = 8

# Request/report then allocate (see module docstring).
CONTROL_ROUNDS = 2


def local_obs_bytes(local_obs_dim: int) -> int:
    return int(local_obs_dim) * B_FLOAT


def profile_bytes(name: str, user_num: int, local_obs_dim: int = 9,
                  num_cells: int = 7):
    """Return (uplink_B, downlink_B, rounds) aggregated over all MDs per slot."""
    U = int(user_num)
    local = local_obs_bytes(local_obs_dim)
    rounds = CONTROL_ROUNDS
    # One broadcast of per-cell compute state for next-slot association.
    cell_bcast = int(num_cells) * B_CELL_COMPUTE
    # Measurement exchange shared by all schemes (fair comparison).
    meas_up = U * B_LOCAL_MEAS
    meas_down = U * B_EDGE_MEAS

    # Distributed MD-local decision (UCB / DTS / Causal / MAPPO):
    # R1 action + learning label + local meas; R2 alloc + edge meas + bcast.
    if name in ("ucb", "dts", "causal", "mappo"):
        up = U * (B_ACTION + B_REWARD) + meas_up
        down = U * B_ALLOC + cell_bcast + meas_down
        return up, down, rounds

    # GDO / SEM-O-RAN: R1 context/offers + local meas; R2 action + alloc + meas.
    if name == "gdo":
        up = U * B_CONTEXT + meas_up
        down = U * (B_ACTION + B_ALLOC) + cell_bcast + meas_down
        return up, down, rounds

    # Centralized joint controller: full local obs + measurement exchange.
    if name in ("cto", "dqn", "ppo"):
        up = U * local + meas_up
        down = U * (B_ACTION + B_ALLOC) + cell_bcast + meas_down
        return up, down, rounds

    # Fallback: decentralized notify + alloc + meas + bcast
    return U * B_ACTION + meas_up, U * B_ALLOC + cell_bcast + meas_down, rounds


def comm_ms_per_slot(name: str, user_num: int, local_obs_dim: int,
                     rtt_s: float, ctrl_rate_bps: float, num_cells: int = 7):
    """Equivalent communication time per slot [ms]."""
    up, down, rounds = profile_bytes(
        name, user_num, local_obs_dim, num_cells=num_cells)
    rate = max(float(ctrl_rate_bps), 1.0)
    t_s = float(rounds) * float(rtt_s) + 8.0 * (up + down) / rate
    return {
        "comm_ms": 1000.0 * t_s,
        "comm_uplink_B": float(up),
        "comm_downlink_B": float(down),
        "comm_rounds": float(rounds),
    }
