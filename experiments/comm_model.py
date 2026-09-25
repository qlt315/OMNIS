"""Control-plane bytes / RTT → equivalent comm time (not in local decision_time)."""

from __future__ import annotations

# --- message payloads [bytes] ---
B_ACTION = 4          # model_idx + cell_rank
B_REWARD = 8          # scalar learning feedback (piggyback next uplink)
B_ALLOC = 16          # bandwidth + GPU (+ MCS ack)
B_CONTEXT = 32        # QoS weights / constraints / coarse SNR
B_FLOAT = 4
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

    # Distributed MD-local decision (UCB / DTS / Causal / MAPPO):
    # R1 action + learning label piggyback; R2 alloc + cell-compute broadcast.
    # Radio (RSRP) is local — not in the byte count.
    if name in ("ucb", "dts", "causal", "mappo"):
        up = U * (B_ACTION + B_REWARD)
        down = U * B_ALLOC + cell_bcast
        return up, down, rounds

    # GDO / SEM-O-RAN: R1 context/offers; R2 ES-chosen action + alloc + bcast.
    if name == "gdo":
        up = U * B_CONTEXT
        down = U * (B_ACTION + B_ALLOC) + cell_bcast
        return up, down, rounds

    # Centralized joint controller: ES already holds compute state; still
    # broadcast so MDs can audit / for a uniform control-plane budget.
    if name in ("cto", "dqn", "ppo"):
        up = U * local
        down = U * (B_ACTION + B_ALLOC) + cell_bcast
        return up, down, rounds

    # Fallback: decentralized notify + alloc + bcast
    return U * B_ACTION, U * B_ALLOC + cell_bcast, rounds


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
