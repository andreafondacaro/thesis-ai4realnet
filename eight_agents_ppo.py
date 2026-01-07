import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

ACTIONS = np.array([-1, +1], dtype=np.int64)


def encode_pair(a, b, max_val=200):
    """Encode a pair (a, b) into a single integer for mi_discrete."""
    return int(a) * (max_val + 1) + int(b)


def mi_discrete(x, y):
    """
    Mutual information I(X;Y) for discrete 1D arrays x, y.
    Uses the exact formula on the empirical joint distribution.
    """
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    assert x.shape == y.shape

    n = x.size

    # relabel values to 0..nx-1 and 0..ny-1
    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    nx, ny = x_vals.size, y_vals.size

    joint = np.zeros((nx, ny), dtype=float)
    for i in range(n):
        joint[x_inv[i], y_inv[i]] += 1.0

    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)   # (nx, 1)
    py = pxy.sum(axis=0, keepdims=True)   # (1, ny)

    mask = pxy > 0
    num = pxy[mask]
    den = (px @ py)[mask]

    return np.sum(num * (np.log(num) - np.log(den)))


def simulate_transitions(
    n_steps=100_000,
    GA_profile=None,
    GB_profile=None,
    seed=0,
):
    """
    Z_t[t]   = (P_L1^t, P_L2^t, P_L3^t,
                G_A^t, G_B^t, A_A^t, A_B^t)       shape (N, 7)

    S_tp1[t] = (P_L1^{t+1}, P_L2^{t+1}, P_L3^{t+1},
                G_A^{t+1}, G_B^{t+1})             shape (N, 5)
    """
    rng = np.random.default_rng(seed)

    if GA_profile is None:
        GA_profile = np.full(n_steps + 1, 100.0, dtype=float)
    else:
        GA_profile = np.asarray(GA_profile, dtype=float)

    if GB_profile is None:
        GB_profile = np.full(n_steps + 1, 100.0, dtype=float)
    else:
        GB_profile = np.asarray(GB_profile, dtype=float)

    assert GA_profile.shape[0] >= n_steps + 1
    assert GB_profile.shape[0] >= n_steps + 1

    max_total = GA_profile.max() + GB_profile.max()
    max_val_for_enc = int(np.ceil(max_total))

    Z_t   = np.zeros((n_steps, 7), dtype=int)
    S_tp1 = np.zeros((n_steps, 5), dtype=int)

    idxA = rng.integers(0, len(L_SPLITS))
    idxB = rng.integers(0, len(L_SPLITS))

    A_L1_pct, A_L2_pct = L_SPLITS[idxA] / 100.0
    B_L2_pct, B_L3_pct = L_SPLITS[idxB] / 100.0

    GA0 = GA_profile[0]
    GB0 = GB_profile[0]

    PL1 = A_L1_pct * GA0
    PL2 = A_L2_pct * GA0 + B_L2_pct * GB0
    PL3 = B_L3_pct * GB0


    for t in range(n_steps):
        GA_tp1 = GA_profile[t + 1]
        GB_tp1 = GB_profile[t + 1]

        PL1_i = int(round(PL1))
        PL2_i = int(round(PL2))
        PL3_i = int(round(PL3))

        GA_state = encode_pair(PL1_i, PL2_i, max_val=max_val_for_enc)
        GB_state = encode_pair(PL2_i, PL3_i, max_val=max_val_for_enc)

        best_cost = np.inf
        best_pairs = []

        target_tp1 = (GA_tp1 + GB_tp1) / 3.0

        for aA in ACTIONS:
            idxA_cand = (idxA + aA) % len(L_SPLITS)
            A_L1_c_pct, A_L2_c_pct = L_SPLITS[idxA_cand] / 100.0

            for aB in ACTIONS:
                idxB_cand = (idxB + aB) % len(L_SPLITS)
                B_L2_c_pct, B_L3_c_pct = L_SPLITS[idxB_cand] / 100.0

                PL1_c = A_L1_c_pct * GA_tp1
                PL2_c = A_L2_c_pct * GA_tp1 + B_L2_c_pct * GB_tp1
                PL3_c = B_L3_c_pct * GB_tp1

                cost = ((PL1_c - target_tp1) ** 2 +
                        (PL2_c - target_tp1) ** 2 +
                        (PL3_c - target_tp1) ** 2)

                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_pairs = [(aA, aB, idxA_cand, idxB_cand, PL1_c, PL2_c, PL3_c)]
                elif abs(cost - best_cost) <= 1e-9:
                    best_pairs.append((aA, aB, idxA_cand, idxB_cand, PL1_c, PL2_c, PL3_c))

        chosen = best_pairs[rng.integers(0, len(best_pairs))]
        AA, AB, idxA_next, idxB_next, PL1_next, PL2_next, PL3_next = chosen

        Z_t[t] = [PL1_i, PL2_i, PL3_i, GA_state, GB_state, AA, AB]

        idxA, idxB = idxA_next, idxB_next

        PL1_next_i = int(round(PL1_next))
        PL2_next_i = int(round(PL2_next))
        PL3_next_i = int(round(PL3_next))

        GA_next_state = encode_pair(PL1_next_i, PL2_next_i, max_val=max_val_for_enc)
        GB_next_state = encode_pair(PL2_next_i, PL3_next_i, max_val=max_val_for_enc)

        S_tp1[t] = [PL1_next_i, PL2_next_i, PL3_next_i, GA_next_state, GB_next_state]

        PL1, PL2, PL3 = PL1_next, PL2_next, PL3_next

    return Z_t, S_tp1, max_val_for_enc


def compute_transition_mi_matrix(Z_t, S_tp1):
    """
    Z_t:   (N, 7) array  [PL1_t, PL2_t, PL3_t, SA_t, SB_t, AA_t, AB_t]
    S_tp1: (N, 5) array  [PL1_tp1, PL2_tp1, PL3_tp1, SA_tp1, SB_tp1]
    Returns: MI matrix of shape (7, 5)
    """
    N, nz = Z_t.shape
    _, ns = S_tp1.shape
    assert nz == 7 and ns == 5

    M = np.zeros((nz, ns))
    for i in range(nz):
        for j in range(ns):
            M[i, j] = mi_discrete(Z_t[:, i], S_tp1[:, j])
    return M


# -----------------------------
# PettingZoo ParallelEnv with dynamic #agents from clusters
# -----------------------------

class PowerChainClusteredParallelEnv(ParallelEnv):
    """
    N generators feeding N+1 adjacent lines (chain).
    Agents = clusters (list of MI-row indices).

    MI conventions:
      - matrix shape = (n_rows, n_cols)
      - "action rows" correspond to controls and live at rows [n_cols .. n_rows-1]
      - n_controls = n_rows - n_cols  (should equal N generators)

    Control mapping:
      - control k corresponds to action-row r = n_cols + k
      - an agent "owns" control k if its cluster contains row r
      - agent's single action is applied to all controls it owns

    Observation:
      - base vector length n_rows
          * rows 0..n_cols-1: state variables at time t (we fill these)
          * rows n_cols..n_rows-1: previous actions per control (we fill these)
        then masked by cluster membership (rows not in cluster -> 0)
      - idx vector length n_controls:
          * idx[k] visible ONLY to the agent that owns control k (else 0)
      - optional forecast: generator outputs at t+1 (length N), visible to all
      - agent id scalar in [0,1] for parameter sharing
    Reward (shared):
      - at t+1, make line loads equal: target = total_generation/(N+1)
      - reward = -sum_j (load_j - target)^2 normalized
    """

    metadata = {"name": "power_chain_clustered_v0"}

    def __init__(
        self,
        G_profiles,          # shape (T, N)
        clusters,            # list[list[int]] of MI row indices
        matrix_shape,        # (n_rows, n_cols) of your MI matrix
        episode_len=1000,
        seed=0,
        include_forecast=True,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.include_forecast = bool(include_forecast)

        G_profiles = np.asarray(G_profiles, dtype=np.float32)
        if G_profiles.ndim != 2:
            raise ValueError("G_profiles must be a 2D array of shape (T, N).")
        self.G = G_profiles
        self.T, self.N = self.G.shape
        self.n_lines = self.N + 1
        self.episode_len = int(episode_len)

        n_rows, n_cols = map(int, matrix_shape)
        if n_rows <= n_cols:
            raise ValueError("matrix_shape must satisfy n_rows > n_cols (need action rows).")
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_controls = n_rows - n_cols

        # Require controls to match generators for this topology
        if self.n_controls != self.N:
            raise ValueError(
                f"Expected n_controls = n_rows-n_cols to equal N generators.\n"
                f"Got n_controls={self.n_controls}, N={self.N}. "
                f"Pass a matrix_shape consistent with your MI construction."
            )

        if self.T < self.episode_len + 2:
            raise ValueError("G_profiles too short for episode_len+2 (needs room for t+1).")

        # Agents = clusters
        self.clusters = [list(map(int, c)) for c in clusters]
        self.n_agents = len(self.clusters)
        if self.n_agents < 1:
            raise ValueError("clusters must not be empty.")

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = []

        # Ownership: each control k is owned by the (first) cluster that contains action-row (n_cols+k)
        self.control_owner = [None] * self.n_controls
        for ai, c in enumerate(self.clusters):
            for k in range(self.n_controls):
                action_row = self.n_cols + k
                if action_row in c and self.control_owner[k] is None:
                    self.control_owner[k] = ai

        # Optional: enforce every control is owned (recommended)
        if any(o is None for o in self.control_owner):
            missing = [k for k, o in enumerate(self.control_owner) if o is None]
            raise ValueError(f"Some controls are unowned by clusters: {missing}. "
                             f"Ensure each action-row {list(range(self.n_cols, self.n_rows))} appears in some cluster.")

        # Scaling
        self.max_total = float(np.max(np.sum(self.G, axis=1)))
        if self.max_total <= 0:
            self.max_total = 1.0

        # Observation size
        # base(n_rows) + idxs(n_controls=N) + forecast(N if on) + agent_id_scalar(1)
        self._obs_dim = self.n_rows + self.n_controls + (self.N if self.include_forecast else 0) + 1

        self._obs_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {a: spaces.Discrete(len(ACTIONS)) for a in self.possible_agents}

        # State
        self.t0 = 0
        self.t = 0
        self.idx = np.zeros(self.N, dtype=np.int64)          # one split index per generator/control
        self.prev_delta = np.zeros(self.N, dtype=np.int64)   # last applied delta per control (e.g. -1/+1)
        self.loads = np.zeros(self.n_lines, dtype=np.float32)

        # Precompute row masks per agent (mask base vector)
        self._base_masks = []
        for c in self.clusters:
            m = np.zeros(self.n_rows, dtype=np.float32)
            for r in c:
                if 0 <= r < self.n_rows:
                    m[r] = 1.0
            self._base_masks.append(m)

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def _compute_loads(self, outputs_t, idx_vec):
        """outputs_t shape (N,), idx_vec shape (N,) -> loads shape (N+1,)"""
        loads = np.zeros(self.n_lines, dtype=np.float32)
        for i in range(self.N):
            left_pct, right_pct = (L_SPLITS[int(idx_vec[i])] / 100.0)
            gi = float(outputs_t[i])
            loads[i]     += left_pct * gi
            loads[i + 1] += right_pct * gi
        return loads

    def _get_obs(self, agent):
        ai = int(agent.split("_")[-1])

        outputs_t = self.G[self.t0 + self.t]          # (N,)
        loads_t = self.loads                          # (N+1,)

        # ---- base vector length n_rows
        base = np.zeros(self.n_rows, dtype=np.float32)

        # Fill "state rows" 0..n_cols-1.
        # Here we define the state variables to be:
        #   [loads (N+1), generator outputs (N)] -> total 2N+1 entries
        # This must match how you built your MI matrix columns.
        state_vec = np.concatenate([loads_t, outputs_t], axis=0)  # length 2N+1
        if state_vec.shape[0] != self.n_cols:
            raise RuntimeError(
                f"Internal mismatch: expected n_cols={self.n_cols} to equal 2N+1={2*self.N+1}.\n"
                f"Either change matrix_shape construction or change state_vec definition."
            )
        base[:self.n_cols] = state_vec / self.max_total

        # Fill "action rows" n_cols..n_rows-1 with previous deltas for each control
        amin = float(np.min(ACTIONS))
        amax = float(np.max(ACTIONS))
        for k in range(self.N):
            r = self.n_cols + k
            a = float(self.prev_delta[k])
            base[r] = 0.0 if amax == amin else (a - amin) / (amax - amin)

        # Mask base by cluster membership
        base *= self._base_masks[ai]

        # ---- idx visibility: only owner sees its idx[k]
        idxs = np.zeros(self.N, dtype=np.float32)
        for k in range(self.N):
            if self.control_owner[k] == ai:
                idxs[k] = float(self.idx[k]) / float(len(L_SPLITS) - 1)

        parts = [base, idxs]

        # ---- optional forecast (outputs at t+1)
        if self.include_forecast:
            outputs_tp1 = self.G[self.t0 + self.t + 1]
            parts.append(outputs_tp1 / self.max_total)

        # ---- agent id scalar (fixed size, works for 4 or 8 agents)
        agent_id = np.array([ai / max(1, self.n_agents - 1)], dtype=np.float32)
        parts.append(agent_id)

        return np.concatenate(parts, axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]

        max_start = self.T - (self.episode_len + 2)
        self.t0 = int(self.rng.integers(0, max_start))
        self.t = 0

        self.idx = self.rng.integers(0, len(L_SPLITS), size=self.N, dtype=np.int64)
        self.prev_delta[:] = 0

        outputs_0 = self.G[self.t0 + self.t]
        self.loads = self._compute_loads(outputs_0, self.idx)

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("step() called on terminated env; call reset().")

        # Apply each agent's action to every control it owns
        for ai in range(self.n_agents):
            ag = f"agent_{ai}"
            a_idx = int(actions[ag])
            delta = int(ACTIONS[a_idx])

            # apply to owned controls
            for k in range(self.N):
                if self.control_owner[k] == ai:
                    self.prev_delta[k] = delta
                    self.idx[k] = (self.idx[k] + delta) % len(L_SPLITS)

        # Compute next loads using outputs at t+1
        outputs_tp1 = self.G[self.t0 + self.t + 1]
        loads_tp1 = self._compute_loads(outputs_tp1, self.idx)

        total_tp1 = float(np.sum(outputs_tp1))
        target = total_tp1 / float(self.n_lines)
        cost = float(np.sum((loads_tp1 - target) ** 2))
        reward = -cost / (self.max_total ** 2)

        self.t += 1
        self.loads = loads_tp1

        env_trunc = (self.t >= self.episode_len)
        terminations = {a: False for a in self.agents}
        truncations  = {a: env_trunc for a in self.agents}
        rewards      = {a: reward for a in self.agents}
        infos        = {a: {} for a in self.agents}

        if env_trunc:
            self.agents = []
            observations = {}
        else:
            observations = {a: self._get_obs(a) for a in self.agents}

        return observations, rewards, terminations, truncations, infos




# -----------------------------
# PPO (parameter sharing) for variable number of agents
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
    agents = env.possible_agents[:]              # dynamic
    n_agents = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]

    model = ActorCritic(obs_dim, n_actions=2).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset()

    ep_ret = 0.0
    ep_rets = []

    for upd in range(total_updates):
        obs_buf  = np.zeros((rollout_steps, n_agents, obs_dim), dtype=np.float32)
        act_buf  = np.zeros((rollout_steps, n_agents), dtype=np.int64)
        logp_buf = np.zeros((rollout_steps, n_agents), dtype=np.float32)
        rew_buf  = np.zeros((rollout_steps, n_agents), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, n_agents), dtype=np.float32)
        val_buf  = np.zeros((rollout_steps, n_agents), dtype=np.float32)

        for t in range(rollout_steps):
            if not env.agents:
                obs, _ = env.reset()

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

            action_dict = {ag: int(act_buf[t, ai]) for ai, ag in enumerate(agents)}
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # shared reward/done -> use first agent as reference
            r = float(rewards[agents[0]])
            d = float(truncations[agents[0]] or terminations[agents[0]])

            rew_buf[t, :] = r
            done_buf[t, :] = d

            ep_ret += r
            if d == 1.0:
                ep_rets.append(ep_ret)
                ep_ret = 0.0

            obs = next_obs

        # bootstrap
        next_vals = np.zeros(n_agents, dtype=np.float32)
        if env.agents:
            for ai, ag in enumerate(agents):
                with torch.no_grad():
                    o = torch.tensor(obs[ag], dtype=torch.float32, device=device).unsqueeze(0)
                    _, v = model(o)
                    next_vals[ai] = float(v.item())
        else:
            next_vals[:] = 0.0

        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        ret_buf = np.zeros_like(rew_buf, dtype=np.float32)
        for ai in range(n_agents):
            adv, ret = gae(rew_buf[:, ai], done_buf[:, ai], val_buf[:, ai], next_vals[ai], gamma=gamma, lam=lam)
            adv_buf[:, ai] = adv
            ret_buf[:, ai] = ret

        B = rollout_steps * n_agents
        obs_f  = obs_buf.reshape(B, obs_dim)
        act_f  = act_buf.reshape(B)
        logp_f = logp_buf.reshape(B)
        adv_f  = adv_buf.reshape(B)
        ret_f  = ret_buf.reshape(B)

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


# -----------------------------
# Example main
# -----------------------------
def discrete_random_state(T, start=100, low=50, high=150, step_choices=(-5, 0, 5), seed=123):
    rng = np.random.default_rng(seed)
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
    if len(returns) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        ma = np.convolve(returns, kernel, mode="valid")
        ma_x = np.arange(window - 1, len(returns))
    else:
        ma = None

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

def find_clusters(matrix):
    clusters = []
    rows, cols = matrix.shape
    for i in range(cols, rows):
        curr_cluster = [i]
        for j in range(cols):
            if matrix[i, j] > 0.5:
                curr_cluster.append(j)
                curr_cluster.append(j + rows)
        clusters.append((curr_cluster))
    return clusters

def influence_fn(matrix, state, action):
    """
    Determines if action influences state based on matrix.

    An action influences a state if matrix[state, action] > 0.5.
    """
    return matrix[action, state - matrix.shape[0]] > 0.5

def _split_states_actions(cluster, n_rows, n_cols):
    """
    Given a cluster of global indices, split into state and action indices.

    cluster: list[int]
    n_states: int == matrix.shape[0]

    Returns:
        states: list[int]   # row indices
        actions: list[int]  # column indices (0-based, no offset)
    """
    states = []
    actions = []
    for idx in cluster:
        if idx > n_cols - 1 and idx < n_rows:
            actions.append(idx)
        elif idx >= n_rows:
            states.append(idx)
    return states, actions


def build_dependencies(matrix, clusters):
    """
    Build a dependency graph between clusters based on influence_fn.

    A cluster i depends on cluster j if there exists:
        state in clusters[i], action in clusters[j]
    such that influence_fn(matrix, state, action) is True.

    Returns:
        dependencies: dict[int, list[int]]  # adjacency list of cluster graph
    """
    n_clusters = len(clusters)

    # Precompute states/actions per cluster
    cluster_states = []
    cluster_actions = []
    for c in clusters:
        s, a = _split_states_actions(c, matrix.shape[0], matrix.shape[1])
        cluster_states.append(s)
        cluster_actions.append(a)

    dependencies = {i: [] for i in range(n_clusters)}

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue

            # i depends on j if one of i's states is influenced by one of j's actions
            depends = False
            for s in cluster_states[i]:
                for a in cluster_actions[j]:
                    if influence_fn(matrix, s, a):
                        depends = True
                        break
                if depends:
                    break

            if depends:
                dependencies[i].append(j)

    return dependencies

def evaluate_matrix(matrix, row_indices, col_indices, shape_cl1, shape_cl2, alpha=0.6):
#computes scores for the 2 separate matrices and compares them to the full matrix
    full_score = 0
    separate_score = 0
    full_matrix = matrix[np.ix_(row_indices, col_indices)]

    for i in range(full_matrix.shape[0]):
        for j in range(full_matrix.shape[1]):
            full_score += alpha * (1 - full_matrix[i, j])
            if i < shape_cl1[0] and j < shape_cl1[1]:
                separate_score += alpha * (1 - full_matrix[i, j])
            elif i >= full_matrix.shape[0] - shape_cl2[0] and j >= full_matrix.shape[1] - shape_cl2[1]:
                separate_score += alpha * (1 - full_matrix[i, j])
            elif (i > shape_cl1[0] and j < full_matrix.shape[1] - shape_cl2[1]) or (i < full_matrix.shape[0] - shape_cl2[0] and j > shape_cl1[1]):
                    separate_score += (1 - alpha) * full_matrix[i, j]
    print(f"Full score: {full_score}, Separate score: {separate_score}")
    

    return separate_score >= full_score, full_score

def find_cycles_dfs(dependencies):
    """
    dependencies: dict[int, list[int]] adjacency list

    Returns:
        cycles: list[list[int]] where each list is a cycle of cluster indices
    """
    visited = set()
    rec_stack = set()
    parent = {}
    cycles = []

    def dfs(u):
        visited.add(u)
        rec_stack.add(u)

        for v in dependencies.get(u, []):
            if v not in visited:
                parent[v] = u
                dfs(v)
            elif v in rec_stack:
                # Found a cycle: extract it
                cycle = [v]
                x = u
                while x != v:
                    cycle.append(x)
                    x = parent[x]
                cycle.reverse()
                cycles.append(cycle)

        rec_stack.remove(u)

    for node in dependencies.keys():
        if node not in visited:
            parent[node] = None
            dfs(node)

    return cycles

def merge_cycles(clusters, cycles, matrix, alpha=0.6, verbose=False):
    """
    Merge clusters according to cycles, but only if evaluate_matrix returns True
    for the candidate merge.

    Assumes:
      - matrix is square
      - cluster elements are indices into matrix rows/cols
    """
    n = len(clusters)
    parent = list(range(n))

    # keep current elements for each DSU root
    comp_elems = {i: set(clusters[i]) for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def try_union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False

        cluster_a = comp_elems[ra]
        cluster_b = comp_elems[rb]

        a_set, b_set = set(cluster_a), set(cluster_b)
        merged = (
        [x for x in cluster_a if x in a_set - b_set] +
        [x for x in cluster_a if x in a_set & b_set] +
        [x for x in cluster_b if x in b_set - a_set]
        )

        # 2) compute row/col indices from merged cluster
        row_indices = [idx for idx in merged if idx < matrix.shape[0]]
        col_indices = [idx - matrix.shape[0] for idx in merged if idx >= matrix.shape[0]]

        shape_a = (len([x for x in cluster_a if x < matrix.shape[0]]),
                    len([x for x in cluster_a if x >= matrix.shape[0]]))
        shape_b = (len([x for x in cluster_b if x < matrix.shape[0]]),
                    len([x for x in cluster_b if x >= matrix.shape[0]]))
        ok, _ = evaluate_matrix(matrix, row_indices, col_indices, shape_a, shape_b, alpha=alpha)
        if ok:
            parent[rb] = ra
            comp_elems[ra].update(comp_elems[rb])
            del comp_elems[rb]
            return True

        return False

    # Apply cycle constraints, but union only when "good"
    for cycle in cycles:
        if not cycle:
            continue
        base = cycle[0]
        for idx in cycle[1:]:
            try_union(base, idx)

    # Return merged clusters (unique roots)
    roots = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, set()).update(clusters[i])

    return [sorted(s) for s in roots.values()]

def merge_clusters(matrix, clusters, alpha):
    """
    Greedy merging of clusters:
    - Always merge the single best pair (lowest score) if evaluate_matrix says it's beneficial.
    - Repeat until no beneficial merge exists (or max_iter is hit).
    - Merged clusters have duplicates removed.
    """
    clusters = [list(set(c)) for c in clusters]

    dependencies = build_dependencies(matrix, clusters)

    # DFS-based cycle detection
    cycles = find_cycles_dfs(dependencies)
    if cycles:
        print("Detected dependency cycles:", cycles)
        clusters = merge_cycles(clusters, cycles, matrix, alpha)
        print("Clusters after merging cycles:", clusters)

    while True:
        best_score = float("inf")
        best_pair = None
        best_merged_cluster = None

        # Find the best pair of clusters to merge
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]

                # 1) merge & dedupe
                a_set, b_set = set(cluster_a), set(cluster_b)
                merged = (
                [x for x in cluster_a if x in a_set - b_set] +
                [x for x in cluster_a if x in a_set & b_set] +
                [x for x in cluster_b if x in b_set - a_set]
                )

                # 2) compute row/col indices from merged cluster
                row_indices = [idx for idx in merged if idx < matrix.shape[0]]
                col_indices = [idx - matrix.shape[0] for idx in merged if idx >= matrix.shape[0]]

                shape_a = (len([x for x in cluster_a if x < matrix.shape[0]]),
                           len([x for x in cluster_a if x >= matrix.shape[0]]))
                shape_b = (len([x for x in cluster_b if x < matrix.shape[0]]),
                           len([x for x in cluster_b if x >= matrix.shape[0]]))

                can_merge, score = evaluate_matrix(matrix, row_indices, col_indices, shape_a, shape_b, alpha)

                if can_merge and score < best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_merged_cluster = merged

        # No beneficial merge found → stop
        if best_pair is None:
            break

        i, j = best_pair
        print(f"Merging clusters {clusters[i]} and {clusters[j]} with score {best_score}")

        new_clusters = []
        for k, c in enumerate(clusters):
            if k == i or k == j:
                continue
            new_clusters.append(c)
        new_clusters.append(best_merged_cluster)

        clusters = new_clusters

    return clusters


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Profiles for training env
    n_steps = 250_000
    T = n_steps + 2

    G_profiles_8 = []
    for i in range(8):
        # vary step sizes a bit so generators aren't identical
        steps = (-5, 0, 5) if i % 2 == 0 else (-10, -5, 0, 5, 10)
        G_profiles_8.append(discrete_random_state(n_steps, start=100, low=50, high=150, step_choices=steps))

    G_profiles_8 = np.stack(G_profiles_8, axis=1)  # shape (T, 8)
 
    
    M = np.array([[0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7],
        [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7]])
    
    alpha = 0.4
    print("MI transition matrix:\n", M)
    clusters = find_clusters(M)
    merged_clusters = merge_clusters(M, clusters, alpha)
    for i in range(len(merged_clusters)):
        merged_clusters[i] = [elem for elem in merged_clusters[i] if elem < M.shape[0]]
    print("Clusters:", clusters, " -> #agents:", len(clusters))

    # ---- 3) build env with dynamic agents
    env = PowerChainClusteredParallelEnv(
        G_profiles=G_profiles_8,
        clusters=merged_clusters,           # your clustering output (row indices)
        matrix_shape=M.shape,
        episode_len=1000,
        seed=29,
        include_forecast=True,
    )

    # ---- 4) train PPO exactly like before (parameter sharing), but with n_agents = len(clusters)
    model, ep_rets = train_ppo_pettingzoo(
        env,
        total_updates=800,
        rollout_steps=2048,
        epochs=10,
        minibatch=256,
        ent_coef=0.03,
        lr=1e-4,
        device=device,
    )

    plot_convergence(ep_rets, window=50, title="Clustered-agent PPO training convergence")
    print("done. episodes logged:", len(ep_rets))
