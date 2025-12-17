# pip install pettingzoo gymnasium torch numpy

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim

from pettingzoo.utils.env import ParallelEnv


# -----------------------------
# Your splits / actions
# -----------------------------
L_SPLITS = np.array([
    [100,   0],
    [ 80,  20],
    [ 60,  40],
    [ 40,  60],
    [ 20,  80],
    [  0, 100],
], dtype=np.float32)

ACTIONS = np.array([-1, 0, +1], dtype=np.int64)


# -----------------------------
# PettingZoo ParallelEnv
# -----------------------------
class PowerSplitParallelEnv(ParallelEnv):
    """
    Two-agent cooperative control:
      - gen_A picks action aA in {-1,0,+1} to change idxA (split A across L1/L2)
      - gen_B picks action aB in {-1,0,+1} to change idxB (split B across L2/L3)

    State transition uses *next* generator outputs GA[t+1], GB[t+1] (your model).
    Reward is shared team reward: -sum_i (PLi_{t+1} - target_{t+1})^2
    """
    metadata = {"name": "power_split_v0"}

    def __init__(self, GA_profile, GB_profile, episode_len=1000, seed=0, include_forecast=True):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.GA = np.asarray(GA_profile, dtype=np.float32)
        self.GB = np.asarray(GB_profile, dtype=np.float32)
        assert self.GA.shape == self.GB.shape
        self.max_total = float(self.GA.max() + self.GB.max())

        self.episode_len = int(episode_len)
        self.include_forecast = bool(include_forecast)

        self.possible_agents = ["gen_A", "gen_B"]
        self.agents = []

        # obs: loads(3) + idxA,idxB(2) + GA,GB(2) + optional GA_next,GB_next(2) + agent_id_onehot(2)
        base = 3 + 2 + 2 + (2 if self.include_forecast else 0) + 2
        self._obs_dim = base

        self._obs_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {a: spaces.Discrete(3) for a in self.possible_agents}

        self.t0 = 0
        self.t = 0
        self.idxA = 0
        self.idxB = 0
        self.PL1 = self.PL2 = self.PL3 = 0.0

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def _get_obs(self, agent):
        # scaled features to help PPO
        loads = np.array([self.PL1, self.PL2, self.PL3], dtype=np.float32) / self.max_total
        idxs  = np.array([self.idxA, self.idxB], dtype=np.float32) / (len(L_SPLITS) - 1)

        GA_t = float(self.GA[self.t0 + self.t])
        GB_t = float(self.GB[self.t0 + self.t])
        cur  = np.array([GA_t, GB_t], dtype=np.float32) / self.max_total

        parts = [loads, idxs, cur]

        if self.include_forecast:
            GA_tp1 = float(self.GA[self.t0 + self.t + 1])
            GB_tp1 = float(self.GB[self.t0 + self.t + 1])
            nxt = np.array([GA_tp1, GB_tp1], dtype=np.float32) / self.max_total
            parts.append(nxt)

        agent_id = np.array([1.0, 0.0], dtype=np.float32) if agent == "gen_A" else np.array([0.0, 1.0], dtype=np.float32)
        parts.append(agent_id)

        return np.concatenate(parts, axis=0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]

        # random starting point so policy generalizes over the profiles
        max_start = len(self.GA) - (self.episode_len + 2)
        self.t0 = int(self.rng.integers(0, max_start))
        self.t = 0

        self.idxA = int(self.rng.integers(0, len(L_SPLITS)))
        self.idxB = int(self.rng.integers(0, len(L_SPLITS)))

        GA0 = float(self.GA[self.t0 + self.t])
        GB0 = float(self.GB[self.t0 + self.t])

        A_L1, A_L2 = (L_SPLITS[self.idxA] / 100.0)
        B_L2, B_L3 = (L_SPLITS[self.idxB] / 100.0)

        self.PL1 = A_L1 * GA0
        self.PL2 = A_L2 * GA0 + B_L2 * GB0
        self.PL3 = B_L3 * GB0

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # actions: dict {agent: 0/1/2} where 0->-1,1->0,2->+1
        if not self.agents:
            raise RuntimeError("step() called on terminated env; call reset().")

        aA = int(actions["gen_A"])
        aB = int(actions["gen_B"])
        dA = int(ACTIONS[aA])
        dB = int(ACTIONS[aB])

        # update split indices
        self.idxA = (self.idxA + dA) % len(L_SPLITS)
        self.idxB = (self.idxB + dB) % len(L_SPLITS)

        # compute next loads using next generator outputs (your model)
        GA_tp1 = float(self.GA[self.t0 + self.t + 1])
        GB_tp1 = float(self.GB[self.t0 + self.t + 1])

        A_L1, A_L2 = (L_SPLITS[self.idxA] / 100.0)
        B_L2, B_L3 = (L_SPLITS[self.idxB] / 100.0)

        PL1_next = A_L1 * GA_tp1
        PL2_next = A_L2 * GA_tp1 + B_L2 * GB_tp1
        PL3_next = B_L3 * GB_tp1

        target = (GA_tp1 + GB_tp1) / 3.0
        cost = (PL1_next - target) ** 2 + (PL2_next - target) ** 2 + (PL3_next - target) ** 2
        reward = -float(cost) / (self.max_total**2)

        # advance time/state
        self.t += 1
        self.PL1, self.PL2, self.PL3 = PL1_next, PL2_next, PL3_next

        env_trunc = (self.t >= self.episode_len)
        terminations = {a: False for a in self.agents}
        truncations  = {a: env_trunc for a in self.agents}
        rewards      = {a: reward for a in self.agents}  # cooperative shared reward
        infos        = {a: {} for a in self.agents}

        if env_trunc:
            # per PettingZoo parallel env creation pattern, clear agents when done :contentReference[oaicite:1]{index=1}
            self.agents = []
            observations = {}
        else:
            observations = {a: self._get_obs(a) for a in self.agents}

        return observations, rewards, terminations, truncations, infos


# -----------------------------
# PPO (parameter sharing)
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)   # logits
        self.v  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.pi(h), self.v(h).squeeze(-1)


def gae(rew, done, val, next_val, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        v_next = next_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * v_next * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    ret = adv + val
    return adv, ret


def train_ppo_pettingzoo(
    env,
    total_updates=400,
    rollout_steps=2048,
    epochs=10,
    minibatch=256,
    lr=3e-4,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    gamma=0.99,
    lam=0.95,
    device="cpu",
):
    obs_dim = env.observation_space("gen_A").shape[0]
    model = ActorCritic(obs_dim, n_actions=3).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset()
    agents = ["gen_A", "gen_B"]

    ep_ret = 0.0
    ep_rets = []

    for upd in range(total_updates):
        # buffers: [T, n_agents, ...]
        obs_buf  = np.zeros((rollout_steps, 2, obs_dim), dtype=np.float32)
        act_buf  = np.zeros((rollout_steps, 2), dtype=np.int64)
        logp_buf = np.zeros((rollout_steps, 2), dtype=np.float32)
        rew_buf  = np.zeros((rollout_steps, 2), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, 2), dtype=np.float32)
        val_buf  = np.zeros((rollout_steps, 2), dtype=np.float32)

        for t in range(rollout_steps):
            # if env ended, reset
            if not env.agents:
                obs, _ = env.reset()

            # policy for both agents (shared parameters)
            for ai, ag in enumerate(agents):
                obs_buf[t, ai] = obs[ag]
                with torch.no_grad():
                    o = torch.tensor(obs[ag], dtype=torch.float32, device=device).unsqueeze(0)
                    logits, v = model(o)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    act_buf[t, ai] = int(a.item())
                    logp_buf[t, ai] = float(dist.log_prob(a).item())
                    val_buf[t, ai] = float(v.item())

            # step parallel env with dict actions
            action_dict = {agents[0]: act_buf[t, 0], agents[1]: act_buf[t, 1]}
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # both agents share same reward/done
            r = float(rewards[agents[0]])
            d = float(truncations[agents[0]] or terminations[agents[0]])

            rew_buf[t, :] = r
            done_buf[t, :] = d

            ep_ret += r
            if d == 1.0:
                ep_rets.append(ep_ret)
                ep_ret = 0.0

            obs = next_obs

        # bootstrap values (if mid-episode)
        next_vals = np.zeros(2, dtype=np.float32)
        if env.agents:
            for ai, ag in enumerate(agents):
                with torch.no_grad():
                    o = torch.tensor(obs[ag], dtype=torch.float32, device=device).unsqueeze(0)
                    _, v = model(o)
                    next_vals[ai] = float(v.item())
        else:
            next_vals[:] = 0.0

        # GAE per agent
        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        ret_buf = np.zeros_like(rew_buf, dtype=np.float32)
        for ai in range(2):
            adv, ret = gae(rew_buf[:, ai], done_buf[:, ai], val_buf[:, ai], next_vals[ai], gamma=gamma, lam=lam)
            adv_buf[:, ai] = adv
            ret_buf[:, ai] = ret

        # flatten across agents: batch = T * 2
        B = rollout_steps * 2
        obs_f  = obs_buf.reshape(B, obs_dim)
        act_f  = act_buf.reshape(B)
        logp_f = logp_buf.reshape(B)
        adv_f  = adv_buf.reshape(B)
        ret_f  = ret_buf.reshape(B)

        # normalize advantages
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        obs_t  = torch.tensor(obs_f, dtype=torch.float32, device=device)
        act_t  = torch.tensor(act_f, dtype=torch.int64, device=device)
        logp_o = torch.tensor(logp_f, dtype=torch.float32, device=device)
        adv_t  = torch.tensor(adv_f, dtype=torch.float32, device=device)
        ret_t  = torch.tensor(ret_f, dtype=torch.float32, device=device)

        idx = np.arange(B)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, B, minibatch):
                mb = idx[start:start + minibatch]
                logits, v = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_o[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                vf_loss = ((v - ret_t[mb]) ** 2).mean()
                loss = pi_loss + vf_coef * vf_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if (upd + 1) % 10 == 0 and len(ep_rets) > 0:
            print(f"update {upd+1}/{total_updates} | last10 mean ep return: {np.mean(ep_rets[-10:]):.3f}")

    return model, ep_rets


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_steps = 250_000
    t = np.arange(n_steps + 1)
    GA_profile = 100 + 80 * np.sin(2 * np.pi * t / 500)
    GB_profile = 100 + 60 * np.cos(2 * np.pi * t / 800)

    env = PowerSplitParallelEnv(GA_profile, GB_profile, episode_len=1000, seed=29, include_forecast=True)

    model, ep_rets = train_ppo_pettingzoo(
        env,
        total_updates=1000,
        rollout_steps=2048,
        epochs=10,
        minibatch=256,
        ent_coef=0.03,
        lr=2e-4,
        device=device,
    )

    print("done. episodes logged:", len(ep_rets))
