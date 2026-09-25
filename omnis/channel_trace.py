"""Uplink channel-gain trace and IRC SINR."""
from __future__ import annotations

import json
import os

import numpy as np


def _as_dict(meta_path):
    if not meta_path or not os.path.isfile(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


class ChannelTrace:
    def __init__(self, npz_path, ue_ids=None, meta=None):
        data = np.load(npz_path)
        H = data["H"]
        if ue_ids is None:
            keep = np.arange(H.shape[2])
        else:
            keep = np.asarray(list(ue_ids), dtype=int)
            if len(keep) != len(set(keep.tolist())):
                raise ValueError(f"ue_ids must be unique: {list(ue_ids)}")
            if keep.min() < 0 or keep.max() >= H.shape[2]:
                raise ValueError("ue_ids not in channel trace")
        self.H = np.ascontiguousarray(H[:, :, keep])
        self.num_slots, self.num_cells, self.num_ues, self.n_rx, self.n_tx = self.H.shape
        self.ue_ids = [int(u) for u in keep]
        self.cell_ids = list(range(self.num_cells))
        self.bandwidth_hz = float(data["bandwidth_hz"])
        self.noise_psd = float(data["noise_psd_w_hz"])
        self.p_max_w = float(data["p_max_w"])
        self.p0_dbm = float(data["p0_dbm"])
        self.alpha = float(data["alpha"])
        self.prb_hz = float(data["prb_hz"])
        self.n_layers = int(data["n_layers"])
        self.eesm_beta = 1.0
        self.meta = dict(meta or {})
        self.meta.setdefault("phy", "channel_gain")
        self.meta.setdefault("rate_layers", self.n_layers)
        self.phy = self.meta.get("phy", "channel_gain")
        self.uplink_state = None
        self.distances = None
        self.sinr_db = self._noise_limited_sinr_db()

    def slot_index(self, t):
        return int(t) % self.num_slots

    def sinr_vector(self, t, ue_local):
        """Noise-limited SINR [dB] from every cell. Used when nobody is transmitting."""
        return self.sinr_db[self.slot_index(t), :, ue_local].copy()

    def sinr(self, t, cell, ue_local):
        return float(self.sinr_db[self.slot_index(t), cell, ue_local])

    def forecast_cell_sinr_db(self, t, ue_local, uplink_state=None):
        """SINR [dB] toward every cell, using the previous slot's transmitters.

        The MD is given the whole cell pool for this forecast, so the ranking
        is the channel, not last slot's share. ``uplink_state`` entries are
        ``(ue_local, cell, bandwidth_hz)``.
        """
        state = [] if uplink_state is None else list(uplink_state)
        out = {}
        bw = float(self.bandwidth_hz)
        for cell in range(self.num_cells):
            sinr_db, _power = self._link_sinr(
                self.slot_index(t), int(ue_local), cell, bw, state)
            out[cell] = sinr_db
        return out

    def realized_radio(self, t, radio):
        """SINR [dB] and transmit power [W] for this slot's radio MDs.

        ``radio`` is a list of ``(ue_local, cell, bandwidth_hz)``. Intra-cell
        users are orthogonal; every entry is an interferer for the others.
        """
        sinr = {}
        power = {}
        slot = self.slot_index(t)
        for ue_local, cell, bw in radio:
            if bw <= 1.0:
                continue
            sinr_db, p_w = self._link_sinr(slot, int(ue_local), int(cell), float(bw), radio)
            sinr[int(ue_local)] = sinr_db
            power[int(ue_local)] = p_w
        return sinr, power

    def _path_loss_db(self, H):
        gain = float(np.sum(np.abs(H) ** 2) / max(self.n_rx * self.n_tx, 1))
        return -10.0 * np.log10(max(gain, 1e-30))

    def _tx_power_w(self, pl_db, bandwidth_hz):
        p_max_dbm = 10.0 * np.log10(self.p_max_w) + 30.0
        prbs = max(float(bandwidth_hz), self.prb_hz) / self.prb_hz
        p_dbm = min(
            p_max_dbm,
            self.p0_dbm + 10.0 * np.log10(prbs) + self.alpha * float(pl_db),
        )
        return float(10.0 ** ((p_dbm - 30.0) / 10.0))

    def _link_sinr(self, slot, ue_local, cell, bandwidth_hz, transmitters):
        H = self.H[slot, cell, ue_local].astype(np.complex128)
        pl = self._path_loss_db(H)
        power = self._tx_power_w(pl, bandwidth_hz)
        cov = self.noise_psd * np.eye(self.n_rx, dtype=np.complex128)
        band = max(self.bandwidth_hz, 1.0)
        for other_ue, other_cell, other_bw in transmitters:
            if int(other_ue) == int(ue_local) or int(other_cell) == int(cell):
                continue
            if other_bw <= 1.0:
                continue
            Hj = self.H[slot, cell, int(other_ue)].astype(np.complex128)
            p_j = self._tx_power_w(self._path_loss_db(
                self.H[slot, int(other_cell), int(other_ue)]), other_bw)
            # Other cell spreads this MD's power across the cell band; the
            # victim sees that PSD on the overlapping fraction b_j/B, which
            # cancels to P_j/B.
            cov += (p_j / band) * (Hj @ Hj.conj().T)
        try:
            whitened = np.linalg.solve(np.linalg.cholesky(cov), H)
        except np.linalg.LinAlgError:
            whitened = np.linalg.solve(cov + 1e-18 * np.eye(self.n_rx), H)
        _u, singular, _vh = np.linalg.svd(whitened, full_matrices=False)
        n_lay = max(1, min(self.n_layers, singular.size))
        # Signal PSD is split across layers. Whitening makes the noise identity,
        # so the layer SINR is the layer power times the squared singular value.
        layer_psd = (power / max(float(bandwidth_hz), 1.0)) / float(n_lay)
        sinr_l = layer_psd * np.square(singular[:n_lay].real)
        sinr_l = np.maximum(sinr_l, 1e-30)
        beta = self.eesm_beta
        sinr_eff = -beta * np.log(np.mean(np.exp(-sinr_l / beta)))
        return float(10.0 * np.log10(max(sinr_eff, 1e-30))), power

    def _noise_limited_sinr_db(self):
        """Cube [slot, cell, ue] with an empty transmitter set and a full-pool grant.

        Noise-limited IRC reduces to the singular values of H. This cube is
        the association reference and the SNR-sweep calibration, not the SINR
        after a loaded slot.
        """
        H = self.H.astype(np.complex128, copy=False)
        gain = np.sum(np.abs(H) ** 2, axis=(-2, -1)) / float(self.n_rx * self.n_tx)
        pl_db = -10.0 * np.log10(np.maximum(gain, 1e-30))
        p_max_dbm = 10.0 * np.log10(self.p_max_w) + 30.0
        prbs = max(self.bandwidth_hz, self.prb_hz) / self.prb_hz
        p_dbm = np.minimum(
            p_max_dbm,
            self.p0_dbm + 10.0 * np.log10(prbs) + self.alpha * pl_db,
        )
        power = 10.0 ** ((p_dbm - 30.0) / 10.0)
        gram = np.einsum("...ri,...rj->...ij", H.conj(), H)
        gram = 0.5 * (gram + np.swapaxes(gram, -1, -2).conj())
        eig = np.linalg.eigvalsh(gram)
        n_lay = max(1, min(self.n_layers, eig.shape[-1]))
        layer_psd = (power / self.bandwidth_hz) / float(n_lay)
        sinr_l = layer_psd[..., None] * np.maximum(eig[..., -n_lay:], 1e-30) / self.noise_psd
        beta = self.eesm_beta
        sinr_eff = -beta * np.log(np.mean(np.exp(-sinr_l / beta), axis=-1))
        return (10.0 * np.log10(np.maximum(sinr_eff, 1e-30))).astype(np.float64)


def load_channel_trace(table_dir, tag="smoke7", ue_ids=None):
    path = os.path.join(table_dir, f"channel_{tag}.npz")
    if not os.path.isfile(path):
        return None
    meta = _as_dict(os.path.join(table_dir, f"channel_{tag}_meta.json"))
    return ChannelTrace(path, ue_ids=ue_ids, meta=meta)
