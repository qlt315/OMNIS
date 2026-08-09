"""Repo-root helpers so PyCharm / IDE runs work without PYTHONPATH tweaks."""
from __future__ import annotations

import os
import sys


def repo_root() -> str:
    """OMNIS repository root (parent of ``experiments/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_repo_root() -> str:
    """``chdir`` to repo root and put it on ``sys.path`` (idempotent)."""
    root = repo_root()
    try:
        if os.path.realpath(os.getcwd()) != os.path.realpath(root):
            os.chdir(root)
    except OSError:
        os.chdir(root)
    exp = os.path.join(root, "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Headless-friendly default when an IDE has no display.
    os.environ.setdefault("MPLBACKEND", "Agg")
    return root
