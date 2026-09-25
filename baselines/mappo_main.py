"""MAPPO: decentralized actors, centralized critic, team Lyapunov reward."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baselines.rl_nets import Critic, LocalActor
from baselines.rl_slot_env import OnlineRLBaseline
from sys_data.config import Config


class MAPPO(OnlineRLBaseline):
    def __init__(self, config):
        if hasattr(config, "mappo_eval"):
            config.rl_eval = bool(config.mappo_eval)
        if hasattr(config, "mappo_dpp_reward"):
            config.rl_dpp_reward = bool(config.mappo_dpp_reward)
        super().__init__(config)
        self.name = "mappo"

        self.gamma = getattr(config, "mappo_gamma", getattr(config, "rl_gamma", 0.99))
        self.gae_lambda = getattr(config, "mappo_gae_lambda", 0.95)
        self.clip_eps = getattr(config, "mappo_clip_eps", 0.2)
        self.entropy_coef = getattr(config, "mappo_entropy_coef", 0.02)
        self.value_coef = getattr(config, "mappo_value_coef", 0.5)
        self.lr = getattr(config, "mappo_lr", 3e-4)
        self.hidden = getattr(config, "mappo_hidden", getattr(config, "rl_hidden", 64))
        self.rollout_len = getattr(config, "mappo_rollout_len", 16)
        self.ppo_epochs = getattr(config, "mappo_epochs", 2)
        self.minibatch_size = getattr(config, "mappo_minibatch_size", 128)
        self.max_grad_norm = getattr(config, "mappo_max_grad_norm", 0.5)

        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.actor = LocalActor(self.local_obs_dim, self.n_actions_local, self.hidden).to(self.device)
        self.critic = Critic(self.global_state_dim, self.hidden).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

        self._buf = {
            "obs": [], "state": [], "actions": [], "logp": [],
            "rew": [], "val": [], "done": [],
        }
        self._pending = None

    def select_actions(self, cand_cells_dic, task_dic, sinr_db_all_dic, t):
        local = np.stack([
            self.local_obs(u, task_dic[u], cand_cells_dic, sinr_db_all_dic)
            for u in self.users
        ], axis=0)
        state = self.global_state(task_dic, cand_cells_dic, sinr_db_all_dic)

        obs_t = torch.from_numpy(local)
        state_t = torch.from_numpy(state)
        with torch.no_grad():
            dist = self.actor.dist(obs_t)
            if self.eval_mode:
                actions = dist.probs.argmax(dim=-1)
            else:
                actions = dist.sample()
            logp = dist.log_prob(actions)
            value = self.critic(state_t)

        actions_np = actions.cpu().numpy().astype(np.int64)
        actions_by_user = {u: int(actions_np[i]) for i, u in enumerate(self.users)}
        self._pending = {
            "obs": local,
            "state": state,
            "actions": actions_np,
            "logp": logp.cpu().numpy().astype(np.float32),
            "val": float(value.item()),
        }
        return self.pack_selections(actions_by_user, cand_cells_dic)

    def _gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_v = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        return adv, adv + values

    def _ppo_update(self, last_value):
        obs = np.stack(self._buf["obs"], axis=0)
        state = np.stack(self._buf["state"], axis=0)
        actions = np.stack(self._buf["actions"], axis=0)
        old_logp = np.stack(self._buf["logp"], axis=0)
        rewards = np.asarray(self._buf["rew"], dtype=np.float32)
        values = np.asarray(self._buf["val"], dtype=np.float32)
        dones = np.asarray(self._buf["done"], dtype=np.float32)

        adv, ret = self._gae(rewards, values, dones, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T, U, obs_dim = obs.shape
        obs_f = obs.reshape(T * U, obs_dim)
        act_f = actions.reshape(T * U)
        logp_f = old_logp.reshape(T * U)
        adv_f = np.repeat(adv, U)

        state_t = torch.tensor(state, dtype=torch.float32)
        ret_t = torch.tensor(ret, dtype=torch.float32)
        obs_t = torch.tensor(obs_f, dtype=torch.float32)
        act_t = torch.tensor(act_f, dtype=torch.int64)
        old_logp_t = torch.tensor(logp_f, dtype=torch.float32)
        adv_t = torch.tensor(adv_f, dtype=torch.float32)

        n = obs_t.shape[0]
        idx = np.arange(n)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                dist = self.actor.dist(obs_t[mb])
                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                slot_ids = mb // U
                v_pred = self.critic(state_t[slot_ids])
                value_loss = nn.functional.mse_loss(v_pred, ret_t[slot_ids])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                self.opt.step()

        for k in self._buf:
            self._buf[k].clear()

    def learn_after_slot(self, slot_info):
        if self._pending is None:
            return
        self._buf["obs"].append(self._pending["obs"])
        self._buf["state"].append(self._pending["state"])
        self._buf["actions"].append(self._pending["actions"])
        self._buf["logp"].append(self._pending["logp"])
        self._buf["rew"].append(float(slot_info["team_r"]))
        self._buf["val"].append(self._pending["val"])
        self._buf["done"].append(float(slot_info["done"]))
        self._pending = None

        t = slot_info["t"]
        done = bool(slot_info["done"])
        if len(self._buf["rew"]) >= self.rollout_len or done:
            with torch.no_grad():
                last_v = 0.0
                if not done:
                    last_v = float(self.critic(
                        torch.from_numpy(slot_info["next_global"])).item())
            self._ppo_update(last_v)

        if t % self.log_every == 0:
            raw = slot_info.get("team_r_raw", slot_info["team_r"])
            print(f'  [mappo {"eval" if self.eval_mode else "train"}] '
                  f'slot={t} team_r={raw:.4f}')


if __name__ == "__main__":
    seed = 0
    config = Config(seed)
    config.update_users(6)
    config.time_slot_num = 40
    agent = MAPPO(config)
    agent.simulation()
    print("aver info:", agent.average_metrics)
    print("action freq:", agent.action_freq)
