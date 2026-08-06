"""Longer queueing+Lyapunov validation: 6 schemes + V sensitivity for causal."""
import time
import numpy as np
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.rss_main import RSS
from baselines.dts_main import DTS
from baselines.cto_main import CTO
from baselines.gdo_main import GDO


def run_one(name, factory, seed=0, slots=100, users=6, lyapunov_v=1.0):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.lyapunov_v = lyapunov_v
    c.cto_max_candidates = 2000
    if name in ('causal', 'ucb'):
        c.algo = name
    t0 = time.time()
    agent = factory(c)
    agent.simulation()
    elapsed = time.time() - t0
    avg = agent.average_metrics
    # second-half metrics to judge convergence / stability
    half = slots // 2
    def half_mean(key):
        vals = []
        for u in agent.users:
            vals.extend(agent.instant_metrics[u][key][half:])
        return float(np.mean(vals)) if vals else float('nan')
    return {
        'name': name, 'V': lyapunov_v,
        'reward': avg['reward'], 'acc': avg['accuracy'],
        'lat': avg['latency'], 'vio': avg['vio_prob'],
        'backlog': avg['backlog_bits'],
        'backlog_2nd': half_mean('backlog'),
        'energy_q': avg['energy_queue'],
        'energy_q_2nd': half_mean('energy_queue'),
        'arrival': avg['arrival_bits'], 'served': avg['served_bits'],
        'sec': elapsed,
    }


def fmt(rows):
    hdr = (f"{'name':8} {'V':>5} {'reward':>8} {'acc':>7} {'lat':>7} {'vio':>7} "
           f"{'Q_avg':>9} {'Q_2nd':>9} {'Z_2nd':>7} {'sec':>6}")
    print(hdr)
    for r in rows:
        print(f"{r['name']:8} {r['V']:5.1f} {r['reward']:8.4f} {r['acc']:7.4f} "
              f"{r['lat']:7.4f} {r['vio']:7.4f} {r['backlog']:9.1f} "
              f"{r['backlog_2nd']:9.1f} {r['energy_q_2nd']:7.3f} {r['sec']:6.1f}")


def main():
    slots, users, seed = 100, 6, 0
    print(f'=== 6-scheme compare (slots={slots}, users={users}, V=1) ===', flush=True)
    jobs = [
        ('causal', OMNIS), ('ucb', OMNIS), ('rss', RSS),
        ('dts', DTS), ('gdo', GDO), ('cto', CTO),
    ]
    rows = []
    for name, cls in jobs:
        print(f'  running {name}...', flush=True)
        rows.append(run_one(name, cls, seed=seed, slots=slots, users=users, lyapunov_v=1.0))
    fmt(rows)

    print(f'\n=== V sensitivity for causal (slots={slots}) ===', flush=True)
    v_rows = []
    for V in (0.1, 1.0, 5.0, 10.0):
        print(f'  running causal V={V}...', flush=True)
        v_rows.append(run_one('causal', OMNIS, seed=seed, slots=slots, users=users, lyapunov_v=V))
    fmt(v_rows)


if __name__ == '__main__':
    main()
