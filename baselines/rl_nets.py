"""Shared neural nets for RL baselines (DQN / PPO / MAPPO)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class BranchingQNet(nn.Module):
    """Shared trunk + per-agent Q heads (branching DQN). Legacy / ablation."""

    def __init__(self, state_dim, n_agents, n_actions, hidden=64):
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
        return torch.stack([head(h) for head in self.heads], dim=1)  # [B, U, A]


class JointQNet(nn.Module):
    """Centralized joint-action Q(s, a_1..a_U), aligned with CTO's joint decision.

    Action encoding: flattened one-hot of all agents' local discrete actions.
    Scores a scalar Q for each (state, joint-action) pair — same decision
    structure as CTO's joint GP over (model, cell_rank)^U, with candidate
    sub-sampling when the joint space is intractable.
    """

    def __init__(self, state_dim, n_agents, n_actions_local, hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions_local = n_actions_local
        self.act_dim = n_agents * n_actions_local
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, actions):
        """actions: [B, U] int64 -> [B, U*A] one-hot."""
        B, U = actions.shape
        flat = actions.reshape(-1)  # [B*U]
        oh = nn.functional.one_hot(flat, num_classes=self.n_actions_local).float()
        return oh.reshape(B, U * self.n_actions_local)

    def forward(self, state, actions):
        """state [B, S], actions [B, U] -> Q [B]."""
        x = torch.cat([state, self.encode(actions)], dim=-1)
        return self.net(x).squeeze(-1)


class BranchingActor(nn.Module):
    """Centralized factored policy: π(a_1..a_U | s) ≈ ∏_i π_i(a_i | s)."""

    def __init__(self, state_dim, n_agents, n_actions, hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_actions) for _ in range(n_agents)
        ])

    def logits(self, state):
        """Return [U, A] for a single state or [B, U, A] for a batch.

        Important: unbatched state must be unsqueezed before the trunk. Otherwise
        ``stack(..., dim=1)`` on per-head ``[A]`` tensors yields ``[A, U]``, which
        breaks when ``U > A`` (e.g. 20 users, 18 local actions).
        """
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        h = self.trunk(state)
        out = torch.stack([head(h) for head in self.heads], dim=1)  # [B, U, A]
        return out.squeeze(0) if single else out

    def dist(self, state):
        # Independent categoricals per agent; returns list of Categorical
        logits = self.logits(state)  # [U, A] or [B, U, A]
        if logits.dim() == 2:
            # [U, A]
            return [Categorical(logits=logits[i]) for i in range(self.n_agents)]
        # batched: per-agent Categorical on [B, A]
        return [Categorical(logits=logits[:, i, :]) for i in range(self.n_agents)]

    def sample(self, state, deterministic=False):
        dists = self.dist(state)
        actions, logps = [], []
        for d in dists:
            a = d.probs.argmax(dim=-1) if deterministic else d.sample()
            actions.append(a)
            logps.append(d.log_prob(a))
        # [U] or [B, U]
        if actions[0].dim() == 0:
            return torch.stack(actions), torch.stack(logps)
        return torch.stack(actions, dim=1), torch.stack(logps, dim=1)

    def evaluate(self, state, actions):
        """actions: [B, U] -> logp [B, U], entropy [B, U]."""
        dists = self.dist(state)
        logps, ents = [], []
        for i, d in enumerate(dists):
            logps.append(d.log_prob(actions[:, i]))
            ents.append(d.entropy())
        return torch.stack(logps, dim=1), torch.stack(ents, dim=1)


class LocalActor(nn.Module):
    """Parameter-shared decentralized actor π(a | o_local)."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)

    def dist(self, obs):
        return Categorical(logits=self.forward(obs))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class RunningMeanStd:
    """Welford running mean/std for reward normalization."""

    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.eps = eps

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (x - self.mean)

    @property
    def std(self):
        if self.count < 2:
            return 1.0
        return float((self.M2 / self.count) ** 0.5) + self.eps

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std
