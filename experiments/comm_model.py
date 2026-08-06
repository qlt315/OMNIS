"""Control-plane communication model for distributed vs centralized schemes.

Wall-clock ``decision_time`` on a single machine does **not** include agent
interaction. This module accounts for per-slot uplink/downlink bytes and
rounds, then converts to an equivalent time:

    T_comm = rounds * RTT + 8 * (uplink + downlink) / R_ctrl

Payload sizes are intentionally simple (order-of-magnitude control messages)
so schemes are comparable; tweak via Config if needed.
"""

from __future__ import annotations

# --- message payloads [bytes] ---
B_ACTION = 4          # model_idx + cell_rank
B_REWARD = 8          # scalar learning feedback
B_ALLOC = 16          # bandwidth + GPU (+ MCS ack)
B_CONTEXT = 32        # QoS weights / constraints / coarse SNR
B_CAUSAL_OBS = 20     # SINR, model, MCS, acc for shared GP
B_GP_BROADCAST = 256  # compressed shared-GP posterior / scores
B_FLOAT = 4


def local_obs_bytes(local_obs_dim: int) -> int:
    return int(local_obs_dim) * B_FLOAT


def profile_bytes(name: str, user_num: int, local_obs_dim: int = 9):
    """Return (uplink_B, downlink_B, rounds) aggregated over all MDs per slot."""
    U = int(user_num)
    local = local_obs_bytes(local_obs_dim)

    # Decentralized execution: local decision, notify ES, receive allocation.
    # Learning feedback (reward) on the uplink for bandit / MAPPO online.
    if name in ("ucb", "dts", "mappo", "rss"):
        up = U * (B_ACTION + B_REWARD)
        down = U * B_ALLOC
        return up, down, 1

    # Causal-shared: pool interventional observations into a shared mechanism GP;
    # ES broadcasts a compact posterior / score summary, then allocations.
    if name == "causal":
        up = U * B_CAUSAL_OBS
        down = U * B_ALLOC + B_GP_BROADCAST
        return up, down, 1

    # GDO / SEM-O-RAN style: MDs report context/offers; ES returns admission + alloc.
    if name == "gdo":
        up = U * B_CONTEXT
        down = U * (B_ACTION + B_ALLOC)
        return up, down, 1

    # Centralized joint controller (CTO / joint-DQN / central PPO):
    # full team state uplink, joint actions + allocations downlink.
    if name in ("cto", "dqn", "ppo"):
        up = U * local
        down = U * (B_ACTION + B_ALLOC)
        return up, down, 1

    # Fallback: treat as decentralized notify
    return U * B_ACTION, U * B_ALLOC, 1


def comm_ms_per_slot(name: str, user_num: int, local_obs_dim: int,
                     rtt_s: float, ctrl_rate_bps: float):
    """Equivalent communication time per slot [ms]."""
    up, down, rounds = profile_bytes(name, user_num, local_obs_dim)
    rate = max(float(ctrl_rate_bps), 1.0)
    t_s = float(rounds) * float(rtt_s) + 8.0 * (up + down) / rate
    return {
        "comm_ms": 1000.0 * t_s,
        "comm_uplink_B": float(up),
        "comm_downlink_B": float(down),
        "comm_rounds": float(rounds),
    }
