"""Short smoke comparison of all 6 schemes under the queueing+Lyapunov framework."""
import time
import numpy as np
from sys_data.config import Config
from omnis.omnis_main import OMNIS
from baselines.rss_main import RSS
from baselines.dts_main import DTS
from baselines.cto_main import CTO
from baselines.gdo_main import GDO


def run(name, factory, seed=0, slots=40, users=6):
    c = Config(seed)
    c.time_slot_num = slots
    c.update_users(users)
    c.cto_max_candidates = 2000  # CTO joint space is huge
    if name == 'causal':
        c.algo = 'causal'
    elif name == 'ucb':
        c.algo = 'ucb'
    t0 = time.time()
    agent = factory(c)
    agent.simulation()
    elapsed = time.time() - t0
    avg = agent.average_metrics
    final_q = np.mean([agent.backlog[u] for u in agent.users])
    final_z = np.mean([agent.energy_queue[u] for u in agent.users])
    return {
        'name': name,
        'reward': avg['reward'],
        'acc': avg['accuracy'],
        'lat': avg['latency'],
        'energy': avg['energy'],
        'vio': avg['vio_prob'],
        'backlog': avg['backlog_bits'],
        'served': avg['served_bits'],
        'arrival': avg['arrival_bits'],
        'final_Q': final_q,
        'final_Z': final_z,
        'sec': elapsed,
    }


def main():
    slots, users, seed = 40, 6, 0
    jobs = [
        ('causal', OMNIS),
        ('ucb', OMNIS),
        ('rss', RSS),
        ('dts', DTS),
        ('gdo', GDO),
        ('cto', CTO),
    ]
    rows = []
    for name, cls in jobs:
        print(f'=== running {name} ===', flush=True)
        rows.append(run(name, cls, seed=seed, slots=slots, users=users))
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in rows[-1].items()}, flush=True)

    print('\n=== SUMMARY ===')
    hdr = f"{'name':8} {'reward':>8} {'acc':>7} {'lat':>7} {'vio':>7} {'backlog':>10} {'Q_end':>10} {'sec':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['name']:8} {r['reward']:8.4f} {r['acc']:7.4f} {r['lat']:7.4f} "
              f"{r['vio']:7.4f} {r['backlog']:10.1f} {r['final_Q']:10.1f} {r['sec']:7.1f}")


if __name__ == '__main__':
    main()
