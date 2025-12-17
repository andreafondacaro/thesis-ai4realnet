import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# -----------------------------
# Problem setup (your splits)
# -----------------------------
L_SPLITS = np.array([
    [100,   0],
    [ 80,  20],
    [ 60,  40],
    [ 40,  60],
    [ 20,  80],
    [  0, 100],
])  # percentages, sum=100

ACTIONS = [-1, 0, +1]  # per-generator action

def action_id_to_pair(aid: int):
    # 0..8 -> (aA,aB)
    i, j = divmod(aid, 3)
    return ACTIONS[i], ACTIONS[j]

# -----------------------------
# Environment
# -----------------------------
class PowerSplitEnv:
    """
    Observation includes:
      - current line loads (PL1,PL2,PL3)
      - current split indices (idxA, idxB)
      - current generator outputs (GA_t, GB_t)
      - next generator outputs (GA_{t+1}, GB_{t+1})  <-- "1-step forecast"
    Reward:
      - negative squared deviation from equal share at t+1
    """
    def __init__(self, GA_profile, GB_profile, episode_len=1000, seed=0):
        self.rng = np.random.default_rng(seed)
        self.GA = np.asarray(GA_profile, dtype=float)
        self.GB = np.asarray(GB_profile, dtype=float)
        assert self.GA.shape == self.GB.shape
        self.T = episode_len
        self.max_total = float(self.GA.max() + self.GB.max())
        self.n_splits = len(L_SPLITS)
        self.reset()

    @property
    def obs_dim(self):
        return 3 + 2 + 2 + 2  # loads + idx + current GA/GB + next GA/GB

    @property
    def n_actions(self):
        return 9

    def _get_obs(self):
        # Scale to ~[0,1] to help learning.
        pl = np.array([self.PL1, self.PL2, self.PL3], dtype=np.float32) / self.max_total
        idx = np.array([self.idxA, self.idxB], dtype=np.float32) / (self.n_splits - 1)
        ga = np.array([self.GA_t, self.GB_t], dtype=np.float32) / self.max_total
        ga_next = np.array([self.GA_tp1, self.GB_tp1], dtype=np.float32) / self.max_total
        return np.concatenate([pl, idx, ga, ga_next], axis=0)

    def reset(self):
        # random phase start so it generalizes across the profile
        self.t0 = int(self.rng.integers(0, len(self.GA) - (self.T + 1)))
        self.t = 0

        self.idxA = int(self.rng.integers(0, self.n_splits))
        self.idxB = int(self.rng.integers(0, self.n_splits))

        self.GA_t = float(self.GA[self.t0 + self.t])
        self.GB_t = float(self.GB[self.t0 + self.t])
        self.GA_tp1 = float(self.GA[self.t0 + self.t + 1])
        self.GB_tp1 = float(self.GB[self.t0 + self.t + 1])

        A_L1, A_L2 = L_SPLITS[self.idxA] / 100.0
        B_L2, B_L3 = L_SPLITS[self.idxB] / 100.0

        self.PL1 = A_L1 * self.GA_t
        self.PL2 = A_L2 * self.GA_t + B_L2 * self.GB_t
        self.PL3 = B_L3 * self.GB_t

        return self._get_obs()

    def step(self, action_id: int):
        aA, aB = action_id_to_pair(int(action_id))

        # candidate next indices
        self.idxA = (self.idxA + aA) % self.n_splits
        self.idxB = (self.idxB + aB) % self.n_splits

        # next outputs
        GA_tp1 = float(self.GA[self.t0 + self.t + 1])
        GB_tp1 = float(self.GB[self.t0 + self.t + 1])

        A_L1, A_L2 = L_SPLITS[self.idxA] / 100.0
        B_L2, B_L3 = L_SPLITS[self.idxB] / 100.0

        # next loads use next outputs (your model)
        PL1_next = A_L1 * GA_tp1
        PL2_next = A_L2 * GA_tp1 + B_L2 * GB_tp1
        PL3_next = B_L3 * GB_tp1

        target = (GA_tp1 + GB_tp1) / 3.0
        cost = (PL1_next - target) ** 2 + (PL2_next - target) ** 2 + (PL3_next - target) ** 2

        reward = -float(cost) / (self.max_total**2)

        # advance time + state
        self.t += 1
        done = (self.t >= self.T)

        self.PL1, self.PL2, self.PL3 = PL1_next, PL2_next, PL3_next
        self.GA_t, self.GB_t = GA_tp1, GB_tp1

        # update "next" for observation (if episode not done)
        if not done:
            self.GA_tp1 = float(self.GA[self.t0 + self.t + 1])
            self.GB_tp1 = float(self.GB[self.t0 + self.t + 1])
        else:
            self.GA_tp1 = self.GA_t
            self.GB_tp1 = self.GB_t

        obs = self._get_obs()
        return obs, reward, done, {}

# -----------------------------
# PPO (discrete) implementation
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
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

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    rollout_steps: int = 2048
    mini_batch: int = 256
    epochs: int = 10

    total_updates: int = 400  # increase for better performance

def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """
    rewards: [T]
    dones:   [T] (1 if done else 0)
    values:  [T]
    next_value: scalar
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret

def train_ppo(env: PowerSplitEnv, cfg=PPOConfig(), device="cpu"):
    obs_dim, n_actions = env.obs_dim, env.n_actions
    model = ActorCritic(obs_dim, n_actions).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    obs = env.reset()
    ep_return = 0.0
    returns_log = []

    for upd in range(cfg.total_updates):
        # rollout storage
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)

        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs

            with torch.no_grad():
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = model(o)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)

            act = int(a.item())
            next_obs, reward, done, _ = env.step(act)

            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            done_buf[t] = 1.0 if done else 0.0
            val_buf[t] = float(v.item())

            ep_return += reward
            obs = next_obs

            if done:
                returns_log.append(ep_return)
                ep_return = 0.0
                obs = env.reset()

        # bootstrap value for last obs
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_v = model(o)
            next_v = float(next_v.item())

        adv, ret = compute_gae(
            rew_buf, done_buf, val_buf, next_v,
            cfg.gamma, cfg.gae_lambda
        )
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # convert to torch
        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        logp_old_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        # PPO updates
        idxs = np.arange(cfg.rollout_steps)
        for _ in range(cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, cfg.rollout_steps, cfg.mini_batch):
                mb = idxs[start:start + cfg.mini_batch]

                logits, v = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                vf_loss = ((v - ret_t[mb]) ** 2).mean()

                loss = pi_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        if (upd + 1) % 10 == 0 and len(returns_log) > 0:
            print(f"Update {upd+1}/{cfg.total_updates} | "
                  f"last 10 ep return avg: {np.mean(returns_log[-10:]):.3f}")

    return model, returns_log

# -----------------------------
# Evaluate a trained policy
# -----------------------------
def evaluate(env: PowerSplitEnv, model: nn.Module, episodes=5, device="cpu"):
    model.eval()
    rets = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(o)
                a = torch.argmax(logits, dim=-1).item()  # greedy
            obs, r, done, _ = env.step(a)
            ep_ret += r
        rets.append(ep_ret)
    return rets

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Your oscillating profiles
    n_steps = 200_000
    t = np.arange(n_steps + 1)
    GA_profile = 100 + 80 * np.sin(2 * np.pi * t / 500)
    GB_profile = 100 + 60 * np.cos(2 * np.pi * t / 800)

    env = PowerSplitEnv(GA_profile, GB_profile, episode_len=1000, seed=29)

    cfg = PPOConfig(
        rollout_steps=2048,
        mini_batch=256,
        epochs=10,
        total_updates=400,  # bump to 1000+ if you want stronger convergence
        lr=3e-4,
    )

    model, train_returns = train_ppo(env, cfg=cfg, device=device)

    test_env = PowerSplitEnv(GA_profile, GB_profile, episode_len=1000, seed=123)
    rets = evaluate(test_env, model, episodes=10, device=device)
    print("Eval returns:", rets)
    print("Eval mean:", float(np.mean(rets)))
