#!/usr/bin/env python3
"""Detach overnight sweeps so Cursor shell exits don't kill the job (macOS-safe)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "sweeps")
LOG = os.path.join(OUT, "overnight.log")
PID = os.path.join(OUT, "overnight.pid")


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"\n===== daemon launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    env["MPLBACKEND"] = "Agg"
    # Full re-run: w_acc=8, V=3, bw=2.4e5, sinr_offset=3, arrival λ∈{0.05..0.40}.
    cmd = [
        sys.executable, "-u",
        os.path.join(ROOT, "experiments", "run_sweeps.py"),
        "--algos", "all",
        "--seeds", "0", "1", "2",
        "--slots", "500",
        "--users", "10",
    ]
    log_f = open(LOG, "a")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open(PID, "w") as f:
        f.write(str(proc.pid))
    print(f"launched pid={proc.pid} log={LOG}")
    time.sleep(25)
    rc = proc.poll()
    if rc is not None:
        print(f"ERROR: process exited early rc={rc}")
        sys.exit(1)
    with open(LOG) as f:
        tail = f.readlines()[-12:]
    print("still running; recent log:")
    print("".join(tail))
    print(f"monitor: tail -f {LOG}")


if __name__ == "__main__":
    main()
