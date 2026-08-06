"""Pretrain the centralized branching DQN baseline and save the checkpoint.

Trains online for ``config.dqn_pretrain_slots`` slots on the same
environment/seed as evaluation, then saves weights to
``config.dqn_model_path`` (default: figures/dqn_pretrained.pt).

Also dumps the training reward curve to figures/dqn_train_curve.csv for the
convergence plot.

Usage:
  PYTHONPATH=. python scripts/train_dqn.py
  PYTHONPATH=. python scripts/train_dqn.py --slots 400 --seed 0
"""

import argparse
import csv
import os

from sys_data.config import Config
from baselines.dqn_main import DQN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, default=None,
                   help="override config.dqn_pretrain_slots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--users", type=int, default=6)
    args = p.parse_args()

    c = Config(args.seed)
    c.update_users(args.users)
    if args.slots is not None:
        c.dqn_pretrain_slots = args.slots
    c.dqn_eval = False

    print(f"pretraining DQN: slots={c.dqn_pretrain_slots} users={args.users} "
          f"seed={args.seed} -> {c.dqn_model_path}", flush=True)
    agent = DQN(c)
    agent.pretrain()

    csv_path = os.path.join(os.path.dirname(c.dqn_model_path) or ".",
                            "dqn_train_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "mean_reward"])
        for t, r in enumerate(agent.train_rewards):
            w.writerow([t, f"{r:.6f}"])
    print(f"saved checkpoint -> {c.dqn_model_path}")
    print(f"saved train curve -> {csv_path} ({len(agent.train_rewards)} slots)")


if __name__ == "__main__":
    main()
