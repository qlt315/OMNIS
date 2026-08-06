"""6-scheme smoke under multi-cell joint (model, cell_rank) arms."""
import time
import numpy as np
from collections import Counter
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.rss_main import RSS
from baselines.dts_main import DTS
from baselines.cto_main import CTO
from baselines.gdo_main import GDO


def cell_hist(agent):
    cells = []
    for u in agent.users:
        cells.extend(agent.instant_metrics[u]["cell"])
    return dict(sorted(Counter(cells).items()))


def run(name, cls, seed=0, slots=40, users=6):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.cto_max_candidates = 2000
    if name in ("causal", "ucb"):
        c.algo = name
    t0 = time.time()
    agent = cls(c)
    agent.simulation()
    elapsed = time.time() - t0
    avg = agent.average_metrics
    return {
        "name": name,
        "reward": avg["reward"],
        "acc": avg["accuracy"],
        "lat": avg["latency"],
        "vio": avg["vio_prob"],
        "backlog": avg["backlog_bits"],
        "cells": cell_hist(agent),
        "sec": elapsed,
    }


def main():
    jobs = [
        ("causal", OMNIS), ("ucb", OMNIS), ("rss", RSS),
        ("dts", DTS), ("gdo", GDO), ("cto", CTO),
    ]
    rows = []
    for name, cls in jobs:
        print(f"=== {name} ===", flush=True)
        r = run(name, cls)
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in r.items()}, flush=True)
    print("\n=== SUMMARY ===")
    print(f"{'name':8} {'reward':>8} {'acc':>7} {'lat':>7} {'vio':>7} "
          f"{'backlog':>10} {'sec':>7}")
    for r in rows:
        print(f"{r['name']:8} {r['reward']:8.4f} {r['acc']:7.4f} {r['lat']:7.4f} "
              f"{r['vio']:7.4f} {r['backlog']:10.1f} {r['sec']:7.1f}")


if __name__ == "__main__":
    main()
