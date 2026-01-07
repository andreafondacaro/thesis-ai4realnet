import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt

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

ACTIONS = np.array([-1, +1], dtype=np.int64)

def action_id_to_pair(aid: int):
    # 0..8 -> (aA,aB)
    i, j = divmod(aid, 2)
    return ACTIONS[i], ACTIONS[j]

def discrete_random_state(T, start=100, low=50, high=150,
                                 step_choices=(-5, 0, 5)):
    rng = np.random.default_rng(123)

    x = np.empty(T, dtype=int)
    x[0] = int(np.clip(start, low, high))

    step_choices = np.array(step_choices, dtype=int)

    for t in range(1, T):
        x[t] = int(np.clip(x[t-1] + rng.choice(step_choices), low, high))

    return x

def plot_convergence(returns_log, window=50, title="Training convergence"):
    if len(returns_log) == 0:
        print("No episode returns logged, nothing to plot.")
        return

    returns = np.asarray(returns_log, dtype=np.float32)

    # Moving average (simple)
    if len(returns) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        ma = np.convolve(returns, kernel, mode="valid")
        ma_x = np.arange(window - 1, len(returns))
    else:
        ma = None
    
    plt.savefig("8_single_ppo_convergence.png", dpi=200)

    plt.figure()
    plt.plot(returns, linewidth=1.0, alpha=0.35, label="Episode return")
    if ma is not None:
        plt.plot(ma_x, ma, linewidth=2.0, label=f"Moving avg ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Return (sum of rewards)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Environment
# -----------------------------

class PowerSplitEnv:
    """
    Single agent controls N generators in a chain feeding N+1 loads.

    Observation:
      loads(N+1) + idxs(N) + G_cur(N) + (G_next(N) if include_forecast)
      scaled by max_total, idxs scaled by (n_splits-1)

    Action:
      single discrete action_id in [0, 3**N - 1]
      interpreted as base-3 digits per generator:
        digit 0/1/2 -> delta -1/0/+1
    Reward:
      - sum_j (load_j(t+1) - target(t+1))^2, normalized by max_total^2
      target(t+1) = sum_i G_i(t+1) / (N+1)
    """

    def __init__(self, G_profiles, episode_len=1000, seed=0, include_forecast=True):
        self.rng = np.random.default_rng(seed)

        self.G = [np.asarray(g, dtype=np.float32) for g in G_profiles]
        self.N = len(self.G)
        assert self.N >= 2
        T0 = len(self.G[0])
        for gi in self.G:
            assert len(gi) == T0

        self.T = int(episode_len)
        self.include_forecast = bool(include_forecast)

        self.n_splits = len(L_SPLITS)
        self.max_total = float(sum(gi.max() for gi in self.G))
        if self.max_total <= 0:
            self.max_total = 1.0

        # internal state
        self.t0 = 0
        self.t = 0
        self.idxs = np.zeros(self.N, dtype=np.int64)
        self.loads = np.zeros(self.N + 1, dtype=np.float32)

        self.reset()

    @property
    def obs_dim(self):
        return (self.N + 1) + self.N + self.N + (self.N if self.include_forecast else 0)

    @property
    def n_actions(self):
        # PPO expects Discrete(n_actions)
        return 2 ** self.N

    def _compute_loads(self, G_vals):
        splits = (L_SPLITS[self.idxs] / 100.0).astype(np.float32)  # [N,2]
        left = splits[:, 0] * G_vals
        right = splits[:, 1] * G_vals

        loads = np.zeros(self.N + 1, dtype=np.float32)
        loads[0] = left[0]
        for k in range(1, self.N):
            loads[k] = right[k - 1] + left[k]
        loads[self.N] = right[self.N - 1]
        return loads

    def _get_obs(self):
        loads = self.loads.astype(np.float32) / self.max_total
        idxs = self.idxs.astype(np.float32) / (self.n_splits - 1)
        G_cur = np.array([self.G[i][self.t0 + self.t] for i in range(self.N)], dtype=np.float32) / self.max_total

        parts = [loads, idxs, G_cur]

        if self.include_forecast:
            G_nxt = np.array([self.G[i][self.t0 + self.t + 1] for i in range(self.N)], dtype=np.float32) / self.max_total
            parts.append(G_nxt)

        return np.concatenate(parts, axis=0)

    def _decode_action(self, action_id: int):
        """
        action_id -> base-3 digits length N -> deltas in {-1,0,+1}
        """
        a = int(action_id)
        digits = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            digits[i] = a % 2
            a //= 2
        deltas = ACTIONS[digits]  # 0/1 -> -1/+1
        return deltas.astype(np.int64)

    def reset(self):
        # ensure we can access t+1 during rollout
        max_start = len(self.G[0]) - (self.T + 2)
        if max_start < 1:
            raise ValueError("Profiles too short for episode_len + 2.")

        self.t0 = int(self.rng.integers(0, max_start))
        self.t = 0

        self.idxs = self.rng.integers(0, self.n_splits, size=self.N, dtype=np.int64)

        G0 = np.array([self.G[i][self.t0 + self.t] for i in range(self.N)], dtype=np.float32)
        self.loads = self._compute_loads(G0)

        return self._get_obs()

    def step(self, action_id: int):
        # update indices from joint action
        deltas = self._decode_action(action_id)
        self.idxs = (self.idxs + deltas) % self.n_splits

        # compute next loads using next generator outputs
        G_tp1 = np.array([self.G[i][self.t0 + self.t + 1] for i in range(self.N)], dtype=np.float32)
        loads_next = self._compute_loads(G_tp1)

        target = float(G_tp1.sum() / (self.N + 1))
        cost = float(((loads_next - target) ** 2).sum())
        reward = -cost / (self.max_total ** 2)

        # advance time
        self.t += 1
        done = (self.t >= self.T)

        self.loads = loads_next
        obs = self._get_obs() if not done else self._get_obs()  # safe either way
        info = {}
        return obs, reward, done, info

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
    lr: float = 1e-4
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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_steps = 100_000
    t = np.arange(n_steps + 1)

    G_profiles_8 = []
    for i in range(8):
        # vary step sizes a bit so generators aren't identical
        steps = (-5, 0, 5) if i % 2 == 0 else (-10, -5, 0, 5, 10)
        G_profiles_8.append(discrete_random_state(n_steps, start=100, low=50, high=150, step_choices=steps))    
    env = PowerSplitEnv(G_profiles_8, episode_len=1000, seed=29, include_forecast=True)

    cfg = PPOConfig(
        rollout_steps=2048,
        mini_batch=256,
        epochs=20,
        total_updates=1600,
        lr=1e-4,
    )

    model, train_returns = train_ppo(env, cfg=cfg, device=device)
    plot_convergence(train_returns, window=50, title="PPO training convergence")


    test_env = PowerSplitEnv(G_profiles_8, episode_len=1000, seed=29, include_forecast=True)
    rets = evaluate(test_env, model, episodes=10, device=device)
    print("Eval returns:", rets)
    print("Eval mean:", float(np.mean(rets)))
