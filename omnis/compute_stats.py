"""EWMA delay/energy / edge-work stats used by MD and ES forecasts."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


class ComputeStats:
    """Per-branch EWMA stats shared across schemes."""

    def __init__(
        self,
        alpha: float = 0.2,
        n0: int = 1,
        prior_local_s: float = 1e-3,
        prior_local_e: float = 5e-4,
        prior_W: float = 5e-3,
        prior_E_norm: float = 5e-3,
        prior_tx_power: float = 0.1,
    ):
        self.alpha = float(alpha)
        self.n0 = int(n0)
        self.prior_local_s = float(prior_local_s)
        self.prior_local_e = float(prior_local_e)
        self.prior_W = float(prior_W)
        self.prior_E_norm = float(prior_E_norm)
        self.prior_tx_power = float(prior_tx_power)
        # (user, model) -> stats
        self._local_s: Dict[Tuple[str, str], float] = {}
        self._local_e: Dict[Tuple[str, str], float] = {}
        self._local_n: Dict[Tuple[str, str], int] = {}
        # model-level local fallback (shared across MDs)
        self._local_model_s: Dict[str, float] = {}
        self._local_model_e: Dict[str, float] = {}
        self._local_model_n: Dict[str, int] = {}
        # model -> edge work stats
        self._W: Dict[str, float] = {}
        self._E_norm: Dict[str, float] = {}
        self._edge_n: Dict[str, int] = {}
        # MD: EWMA of observed uplink grants [Hz]
        self._bw_grant: Dict[str, float] = {}
        self._bw_grant_n: Dict[str, int] = {}

    def _ewma(self, old: Optional[float], sample: float) -> float:
        if old is None:
            return float(sample)
        a = self.alpha
        return (1.0 - a) * float(old) + a * float(sample)

    def update_from_task(self, task) -> None:
        """Ingest one completed task's measured stage overheads."""
        model = getattr(task, "model_name", None)
        user = getattr(task, "user", None)
        if not model or not user:
            return
        key = (str(user), str(model))
        m = str(model)

        self._local_s[key] = self._ewma(
            self._local_s.get(key), float(task.local_s))
        self._local_e[key] = self._ewma(
            self._local_e.get(key), float(task.local_e))
        self._local_n[key] = int(self._local_n.get(key, 0)) + 1
        self._local_model_s[m] = self._ewma(
            self._local_model_s.get(m), float(task.local_s))
        self._local_model_e[m] = self._ewma(
            self._local_model_e.get(m), float(task.local_e))
        self._local_model_n[m] = int(self._local_model_n.get(m, 0)) + 1

        f_bar = float(getattr(task, "edge_f_bar", 0.0) or 0.0)
        if f_bar <= 1e-12:
            f_bar = 1.0
        W = float(task.edge_s) * f_bar
        E_norm = float(task.edge_e) / max(f_bar * f_bar, 1e-18)
        self._W[m] = self._ewma(self._W.get(m), W)
        self._E_norm[m] = self._ewma(self._E_norm.get(m), E_norm)
        self._edge_n[m] = int(self._edge_n.get(m, 0)) + 1

    def update_bw_grant(self, user: str, bw_hz: float) -> None:
        """Observe one positive radio grant for MD bandwidth prior."""
        if float(bw_hz) <= 1.0:
            return
        u = str(user)
        self._bw_grant[u] = self._ewma(self._bw_grant.get(u), float(bw_hz))
        self._bw_grant_n[u] = int(self._bw_grant_n.get(u, 0)) + 1

    def bw_grant_hat(self, user: str) -> Optional[float]:
        u = str(user)
        if int(self._bw_grant_n.get(u, 0)) >= 1:
            return float(self._bw_grant[u])
        return None

    def local_hat(self, user: str, model: str) -> Tuple[float, float]:
        key = (str(user), str(model))
        n = int(self._local_n.get(key, 0))
        if n >= self.n0:
            return float(self._local_s[key]), float(self._local_e[key])
        m = str(model)
        n_m = int(self._local_model_n.get(m, 0))
        if n_m >= self.n0:
            return float(self._local_model_s[m]), float(self._local_model_e[m])
        if n >= 1:
            return float(self._local_s[key]), float(self._local_e[key])
        if n_m >= 1:
            return float(self._local_model_s[m]), float(self._local_model_e[m])
        return self.prior_local_s, self.prior_local_e

    def edge_work(self, model: str) -> Tuple[float, float]:
        """Return (W_hat, E_norm_hat) for GPU allocation / edge hats."""
        m = str(model)
        n = int(self._edge_n.get(m, 0))
        if n >= self.n0:
            return float(self._W[m]), float(self._E_norm[m])
        if n >= 1:
            return float(self._W[m]), float(self._E_norm[m])
        return self.prior_W, self.prior_E_norm

    def edge_hat(self, model: str, f_hat: float) -> Tuple[float, float]:
        W, E_norm = self.edge_work(model)
        f = max(float(f_hat), 1e-12)
        return W / f, E_norm * (f ** 2)


def ensure_compute_stats(agent) -> ComputeStats:
    """Attach a shared ComputeStats instance on the agent if missing."""
    stats = getattr(agent, "compute_stats", None)
    if stats is None:
        cfg = getattr(agent, "config", None)
        default_p = 0.1
        try:
            users = getattr(agent, "users", None) or []
            if users:
                default_p = float(agent.md_params[users[0]]["trans_power"])
        except Exception:
            pass
        stats = ComputeStats(
            alpha=float(getattr(cfg, "compute_stats_alpha", 0.2)),
            n0=int(getattr(cfg, "compute_stats_n0", 1)),
            prior_local_s=float(getattr(cfg, "compute_prior_local_s", 1e-3)),
            prior_local_e=float(getattr(cfg, "compute_prior_local_e", 5e-4)),
            prior_W=float(getattr(cfg, "compute_prior_W", 5e-3)),
            prior_E_norm=float(getattr(cfg, "compute_prior_E_norm", 5e-3)),
            prior_tx_power=float(getattr(cfg, "compute_prior_tx_power", default_p)),
        )
        agent.compute_stats = stats
    return stats


def predict_service_overheads(
    agent,
    user: str,
    model_name: str,
    mcs_idx: int,
    snr_db: float = 0.0,
    cell_id=None,
):
    """Shared MD decision-time (service, sojourn, energy) without FLOPs."""
    from omnis.exog import md_pending_wait
    from omnis.radio_obs import radio_grant_wait

    stats = ensure_compute_stats(agent)
    local_d, local_e = stats.local_hat(user, model_name)

    gpu_hat = float(agent._gpu_hat_for_association(user, cell_id=cell_id))
    edge_d, edge_e = stats.edge_hat(model_name, gpu_hat)

    bw_fn = getattr(agent, "_forecast_uplink_bw", None)
    if bw_fn is not None:
        bw = float(bw_fn(
            user, cell_id,
            agent._last_bandwidth.get(
                user, agent.total_bandwidth / max(agent.user_num, 1))))
    else:
        bw = float(agent._last_bandwidth.get(
            user, agent.total_bandwidth / max(agent.user_num, 1)))
    se_eff = max(agent.mcs_table.delay_se(model_name, mcs_idx, snr_db), 1e-12)
    bits = float(agent._payload_bits(model_name))
    trans_d = bits / max(bw * se_eff, 1e-12)
    grant = radio_grant_wait(local_d, agent.slot_duration)

    service = local_d + grant + trans_d + edge_d
    pipe = getattr(agent, "pipeline", None)
    md_wait = md_pending_wait(pipe, user) if pipe is not None else 0.0
    sojourn = service + md_wait
    p_tx = tx_power_forecast(agent, user, cell_id=cell_id, bw_hz=bw)
    energy = local_e + p_tx * trans_d + edge_e
    return service, sojourn, energy


def tx_power_forecast(agent, user: str, cell_id=None, bw_hz: float = 0.0) -> float:
    """Fixed uplink power from Config (no power control / OLPC)."""
    del cell_id, bw_hz
    return float(agent.md_params[user]["trans_power"])


def local_edge_compute_parts(agent, user: str, model_name: str, gpu_hat: float):
    """Return (local_d, local_e, edge_d, edge_e) from online stats."""
    stats = ensure_compute_stats(agent)
    local_d, local_e = stats.local_hat(user, model_name)
    edge_d, edge_e = stats.edge_hat(model_name, max(float(gpu_hat), 1e-12))
    return local_d, local_e, edge_d, edge_e
