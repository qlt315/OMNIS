"""Shared radio observation helpers (true vs estimated SINR)."""

from __future__ import annotations


def observe_cell_sinr_db(sinr_trace, time_slot, ue_local, sinr_offset_db=0.0,
                         est_err_db=0.0):
    """Per-cell true and estimated SINR [dB] for one UE at a slot.

    True = trace + offset (environment Acc / BLER / goodput).
    Est  = true + N(0, est_err_db^2) for association and MD decisions.
    """
    sinr_vec = sinr_trace.sinr_vector(time_slot, ue_local)
    true_db = {}
    est_db = {}
    for cell_idx in range(sinr_trace.num_cells):
        tdb = float(sinr_vec[cell_idx]) + float(sinr_offset_db)
        true_db[cell_idx] = tdb
        if est_err_db > 0:
            import numpy as np
            est_db[cell_idx] = tdb + float(est_err_db) * float(np.random.randn())
        else:
            est_db[cell_idx] = tdb
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
