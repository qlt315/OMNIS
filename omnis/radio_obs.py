"""Shared radio observation helpers (true vs estimated SINR)."""

from __future__ import annotations


def observe_cell_sinr_db(sinr_trace, time_slot, ue_local, sinr_offset_db=0.0,
                         est_err_db=0.0, seed=0):
    """Per-cell true and estimated SINR [dB] for one UE at a slot.

    A channel-gain trace forecasts SINR from the previous slot's transmitters.
    An older SINR CSV is read directly. Est = true + keyed N(0, est_err_db^2)
    so CSI noise is identical across algorithms for the same seed/slot/UE.
    """
    from omnis.exog import est_noise_db

    forecast = getattr(sinr_trace, "forecast_cell_sinr_db", None)
    if forecast is not None:
        raw = forecast(
            time_slot, ue_local, getattr(sinr_trace, "uplink_state", None))
        true_db = {
            cell_idx: float(raw[cell_idx]) + float(sinr_offset_db)
            for cell_idx in range(sinr_trace.num_cells)
        }
    else:
        sinr_vec = sinr_trace.sinr_vector(time_slot, ue_local)
        true_db = {}
        for cell_idx in range(sinr_trace.num_cells):
            true_db[cell_idx] = float(sinr_vec[cell_idx]) + float(sinr_offset_db)
    est_db = {}
    for cell_idx, tdb in true_db.items():
        est_db[cell_idx] = tdb + est_noise_db(
            seed, time_slot, ue_local, cell_idx, est_err_db)
    return true_db, est_db


def linear_snr_from_db_map(cell_dic, sinr_db_all_dic, users):
    """Map each user's associated cell SINR [dB] → linear SNR."""
    snr_dic = {}
    for user in users:
        snr_db = sinr_db_all_dic[user][cell_dic[user]]
        snr_dic[user] = 10 ** (snr_db / 10)
    return snr_dic


def payload_bits(data_size_bytes, model_name):
    """On-air payload [bits] from Config.data_size [bytes]."""
    return 8.0 * float(data_size_bytes[model_name])


def radio_grant_wait(local_d, slot_duration):
    """Time from the end of local compute until the next radio grant.

    Bandwidth is assigned at the slot boundary. A task that is still local
    at that boundary cannot transmit until the next slot, and the simulator
    charges that wait as airtime.
    """
    slot = float(slot_duration)
    local_d = float(local_d)
    if local_d <= 1e-12 or slot <= 0.0:
        return 0.0
    frac = local_d % slot
    if frac <= 1e-9:
        return 0.0
    return slot - frac
