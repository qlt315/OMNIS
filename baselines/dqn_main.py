"""Strictly centralized online DQN (branching / multi-head).

Centralized properties (vs previous independent DQN):
  - One learner sees the **global state** (concat of all MD features)
  - One **team reward** R = mean_u (V·r_u + drift_u) per slot
  - One joint decision per slot from a single forward pass

Action space: flat joint size A^U with A=n_models·L is intractable (18^6).
We use **branching DQN** (Tavakoli et al.): shared trunk + U discrete heads.
Heads are optimized jointly against the shared team TD target, so selection
is coordinated through global state + global reward (centralized), not
independent per-MD learners.

Protocol: online-from-scratch over ``time_slot_num`` (ε-greedy + TD).
Optional ``dqn_eval=True`` loads a frozen checkpoint for ablation.
"""

from __future__ import annotations

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baselines.rl_slot_env import OnlineRLBaseline
from sys_data.config import Config


class _BranchingQNet(nn.Module):
    """Shared trunk + per-agent action heads (branching DQN)."""

    def __init__(self, state_dim, n_agents, n_actions, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_actions) for _ in range(n_agents)
        ])

    def forward(self, x):
        h = self.trunk(x)
        # list of [B, A] -> stack [B, U, A]
        return torch.stack([head(h) for head in self.heads], dim=1)


class DQN(OnlineRLBaseline):
    def __init__(self, config):
        # Map legacy dqn_* eval flags onto rl_* used by the shared loop
        if hasattr(config, "dqn_eval"):
            config.rl_eval = bool(config.dqn_eval)
        if hasattr(config, "dqn_dpp_reward"):
            config.rl_dpp_reward = bool(config.dqn_dpp_reward)
        super().__init__(config)
        self.name = "dqn"

        self.gamma = getattr(config, "dqn_gamma", 0.95)
        self.batch_size = getattr(config, "dqn_batch_size", 32)
        self.buffer_size = getattr(config, "dqn_buffer_size", 5000)
        self.lr = getattr(config, "dqn_lr", 1e-3)
        self.target_sync = getattr(config, "dqn_target_sync", 20)
        self.eps_start = getattr(config, "dqn_eps_start", 1.0)
        self.eps_end = getattr(config, "dqn_eps_end", 0.05)
        self.eps_decay_slots = getattr(config, "dqn_eps_decay_slots", 80)
        self.train_every = getattr(config, "dqn_train_every", 1)
        self.grad_steps = getattr(config, "dqn_grad_steps", 4)
        self.hidden = getattr(config, "dqn_hidden", 128)
        self.pretrain_slots = getattr(config, "dqn_pretrain_slots", 400)
        self.model_path = getattr(config, "dqn_model_path", "figures/dqn_pretrained.pt")

        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.q = _BranchingQNet(
            self.global_state_dim, self.user_num, self.n_actions_local, self.hidden
        ).to(self.device)
        self.q_tgt = _BranchingQNet(
            self.global_state_dim, self.user_num, self.n_actions_local, self.hidden
        ).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=self.lr)
        self.buffer = deque(maxlen=self.buffer_size)
        self._step = 0
        self._pending = None  # (global_state, actions[U])

        if self.eval_mode:
            self._load_pretrained()

    def _load_pretrained(self):
        if os.path.exists(self.model_path):
            self.q.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.q_tgt.load_state_dict(self.q.state_dict())
        else:
            raise FileNotFoundError(
                f"DQN eval requested but no pretrained model at {self.model_path}. "
                "Run scripts/train_dqn.py first.")

    def save_pretrained(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        torch.save(self.q.state_dict(), self.model_path)

    def pretrain(self):
        was_eval = self.eval_mode
        prev_slots = self.time_slot_num
        self.eval_mode = False
        self.time_slot_num = self.pretrain_slots
        self._run_simulation()
        self.save_pretrained()
        self.eval_mode = was_eval
        self.time_slot_num = prev_slots
        return self.train_rewards

    def _epsilon(self, t):
        if self.eval_mode:
            return 0.0
        if self.eps_decay_slots <= 0:
            return self.eps_end
        frac = min(1.0, t / float(self.eps_decay_slots))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_actions(self, cand_cells_dic, task_dic, sinr_db_all_dic, t):
        s = self.global_state(task_dic, cand_cells_dic, sinr_db_all_dic)
        eps = self._epsilon(t)
        if random.random() < eps:
            actions = [random.randrange(self.n_actions_local) for _ in self.users]
        else:
            with torch.no_grad():
                qv = self.q(torch.from_numpy(s).unsqueeze(0))  # [1, U, A]
                actions = qv.argmax(dim=2).squeeze(0).tolist()
        actions_by_user = {u: actions[i] for i, u in enumerate(self.users)}
        self._pending = (s, np.asarray(actions, dtype=np.int64))
        return self.pack_selections(actions_by_user, cand_cells_dic)

    def _train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        a = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.int64)  # [B, U]
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        q_all = self.q(s)  # [B, U, A]
        q_sa = q_all.gather(2, a.unsqueeze(2)).squeeze(2)  # [B, U]
        q_team = q_sa.mean(dim=1, keepdim=True)  # match team_r scale
        with torch.no_grad():
            max_next = self.q_tgt(s2).max(dim=2).values  # [B, U]
            max_next_team = max_next.mean(dim=1, keepdim=True)
            target = r + self.gamma * (1.0 - done) * max_next_team
        loss = nn.functional.mse_loss(q_team, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

    def learn_after_slot(self, slot_info):
        if self._pending is None:
            return
        s, actions = self._pending
        s2 = slot_info["next_global"]
        r = float(slot_info["team_r"])
        done = float(slot_info["done"])
        self.buffer.append((s, actions, r, s2, done))
        self._pending = None

        self._step += 1
        if self._step % self.train_every == 0:
            for _ in range(self.grad_steps):
                self._train_step()
        if self._step % self.target_sync == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        t = slot_info["t"]
        if t % 20 == 0:
            print(f'  [dqn train] slot={t} team_r={r:.4f} '
                  f'eps={self._epsilon(t):.3f}')


if __name__ == "__main__":
    seed = 0
    config = Config(seed)
    config.update_users(6)
    config.time_slot_num = 40
    agent = DQN(config)
    agent.simulation()
    print("aver info:", agent.average_metrics)
    print("action freq:", agent.action_freq)
