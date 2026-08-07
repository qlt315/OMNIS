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
    # Truncate log for a clean overnight start marker
    with open(LOG, "a") as f:
        f.write(f"\n===== daemon launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    env["MPLBACKEND"] = "Agg"
    cmd = [
        sys.executable, "-u",
        os.path.join(ROOT, "experiments", "run_sweeps.py"),
        "--algos", "causal", "ucb", "gdo", "dqn", "ppo",
        "--seeds", "0", "1", "2",
        "--slots", "400",
        "--users", "10",
        # Full re-run after GDO Acc-floor restore + users axis 5..25 (no --resume).
    ]
    log_f = open(LOG, "a")
    # start_new_session=True == setsid; works on macOS via Python
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
    time.sleep(20)
    rc = proc.poll()
    if rc is not None:
        print(f"ERROR: process exited early rc={rc}")
        sys.exit(1)
    # Confirm still alive and log growing
    with open(LOG) as f:
        tail = f.readlines()[-8:]
    print("still running; recent log:")
    print("".join(tail))
    print(f"monitor: tail -f {LOG}")


if __name__ == "__main__":
    main()
