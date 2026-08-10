"""Shared helpers to persist experiment results for Python and MATLAB."""
from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.io import savemat


# Keys that plot/replot must never treat as disposable source data.
PYTHON_SOURCE_FILES = (
    "perseed.csv",
    "summary.csv",
    "series",  # directory
)


def save_mat_and_python(path, payload, *, long_field_names=True):
    """Write ``path`` (.mat) and a Python twin ``*.pkl`` (+ flat ``*.npz`` when possible).

    Canonical online outputs remain ``perseed.csv`` / ``series/*.npz``; these
    sidecars are aggregated exports so replot/MATLAB never become the only copy.
    """
    path = os.path.abspath(path)
    if not path.endswith(".mat"):
        path = path + ".mat"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    savemat(path, payload, long_field_names=long_field_names, do_compression=True)

    stem = path[: -len(".mat")]
    pkl_path = stem + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Flat numeric/object arrays also as npz (skip nested dicts — those stay in pkl).
    npz_payload = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            continue
        try:
            npz_payload[k] = np.asarray(v)
        except (TypeError, ValueError):
            continue
    npz_path = stem + ".npz"
    if npz_payload:
        np.savez_compressed(npz_path, **npz_payload)
    return path, pkl_path, npz_path if npz_payload else None


def load_mat_extras(mat_path, *, prefixes=("pick_",), keys=("pick_models", "pick_axis")):
    """Load preservable extras from an existing ``.mat`` (or twin ``.pkl``)."""
    extra = {}
    stem = mat_path[: -len(".mat")] if mat_path.endswith(".mat") else mat_path
    pkl_path = stem + ".pkl"
    src = None
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                src = pickle.load(f)
        except Exception:
            src = None
    if src is None and os.path.isfile(mat_path):
        try:
            from scipy.io import loadmat
            src = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception:
            src = None
    if not isinstance(src, dict):
        return extra
    for k, v in src.items():
        if str(k).startswith("_"):
            continue
        if any(str(k).startswith(p) for p in prefixes) or k in keys:
            extra[k] = v
    return extra
