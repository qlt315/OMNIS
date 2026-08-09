"""Control-plane communication model for distributed vs centralized schemes.

Wall-clock ``decision_time`` on a single machine does **not** include agent
interaction. This module accounts for per-slot uplink/downlink bytes and
control rounds, then converts to an equivalent time on a shared control
channel (TDMA within a round):

    T_comm = rounds * RTT + 8 * (uplink_B + downlink_B) / R_ctrl

Protocol (every scheme, every slot after MD-side compute):
  Round 1 — each MD → ES: report (context / obs / joint state / action)
  ES runs BCD (+ joint decision for centralized)  [not in T_comm]
  Round 2 — ES → each MD: action (if ES-chosen) + resource allocation

So ``rounds = 2`` (request/report + allocate). Learning feedback for online
bandits / MAPPO piggybacks on the next slot's Round-1 uplink.

Byte accounting:
  - Per-MD payloads are multiplied by U (shared control channel → TDMA).
  - One-shot broadcasts (e.g. Causal shared-GP summary) are counted once.
  - Centralized joint controllers upload a larger local observation vector;
    distributed MDs that decide locally only upload a compact action / obs.

Reported ``decision_ms`` for factorized / multi-agent schemes (causal, ucb,
dts, mappo, …) is **parallel** compute: max over agents, not the sequential
sum. Centralized joint methods (cto, dqn, ppo) keep true joint wall.
"""

from __future__ import annotations

# --- message payloads [bytes] ---
B_ACTION = 4          # model_idx + cell_rank
B_REWARD = 8          # scalar learning feedback (piggyback next uplink)
B_ALLOC = 16          # bandwidth + GPU (+ MCS ack)
B_CONTEXT = 32        # QoS weights / constraints / coarse SNR
B_CAUSAL_OBS = 20     # SINR, model, MCS, acc for shared GP
B_GP_BROADCAST = 256  # compressed shared-GP posterior / scores (once / slot)
B_FLOAT = 4

# Request/report then allocate (see module docstring).
CONTROL_ROUNDS = 2


def local_obs_bytes(local_obs_dim: int) -> int:
    return int(local_obs_dim) * B_FLOAT


def profile_bytes(name: str, user_num: int, local_obs_dim: int = 9):
    """Return (uplink_B, downlink_B, rounds) aggregated over all MDs per slot."""
    U = int(user_num)
    local = local_obs_bytes(local_obs_dim)
    rounds = CONTROL_ROUNDS

    # Distributed MD-local decision: R1 action (+ reward piggyback), R2 alloc.
    if name in ("ucb", "dts", "mappo", "rss"):
        up = U * (B_ACTION + B_REWARD)
        down = U * B_ALLOC
        return up, down, rounds

    # Causal-shared: R1 interventional obs to pool GP; R2 alloc + one GP summary.
    if name == "causal":
        up = U * B_CAUSAL_OBS
        down = U * B_ALLOC + B_GP_BROADCAST
        return up, down, rounds

    # GDO / SEM-O-RAN: R1 context/offers; R2 ES-chosen action + alloc.
    if name == "gdo":
        up = U * B_CONTEXT
        down = U * (B_ACTION + B_ALLOC)
        return up, down, rounds

    # Centralized joint controller: R1 full local state; R2 joint action + alloc.
    if name in ("cto", "dqn", "ppo"):
        up = U * local
        down = U * (B_ACTION + B_ALLOC)
        return up, down, rounds

    # Fallback: decentralized notify + alloc
    return U * B_ACTION, U * B_ALLOC, rounds


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
