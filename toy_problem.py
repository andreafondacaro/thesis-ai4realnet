import numpy as np

def discrete_random_state(T, start=100, low=50, high=150,
                                 step_choices=(-5, 0, 5)):
    rng = np.random.default_rng(123)

    x = np.empty(T, dtype=int)
    x[0] = int(np.clip(start, low, high))

    step_choices = np.array(step_choices, dtype=int)

    for t in range(1, T):
        x[t] = int(np.clip(x[t-1] + rng.choice(step_choices), low, high))

    return x

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


rng = np.random.default_rng(29)

L_SPLITS = np.array([
    [100,   0],
    [ 80,  20],
    [ 60,  40],
    [ 40,  60],
    [ 20,  80],
    [  0, 100],
])  # now interpreted as % splits (sum = 100)


def encode_pair(a, b, max_val=200):
    """Encode a pair (a, b) into a single integer for mi_discrete."""
    return int(a) * (max_val + 1) + int(b)


def simulate_transitions(
    n_steps=100_000,
    GA_profile=None,
    GB_profile=None,
):
    """
    Z_t[t]   = (P_L1^t, P_L2^t, P_L3^t,
                G_A^t, G_B^t, A_A^t, A_B^t)       shape (N, 7)

    S_tp1[t] = (P_L1^{t+1}, P_L2^{t+1}, P_L3^{t+1},
                G_A^{t+1}, G_B^{t+1})             shape (N, 5)

    Now:
      - L_SPLITS rows are percentages of each generator's output.
      - GA_profile[t], GB_profile[t] give the *time-varying* output
        of generator A and B at time t (same units for all lines).

    Generators are always "on":
      - Generator A outputs GA_profile[t], split across L1 and L2.
      - Generator B outputs GB_profile[t], split across L2 and L3.

    Cost objective still: make (P_L1, P_L2, P_L3) as equal as possible
    at t+1, given next-step generator outputs.
    """

    # Default: recover original behavior (both fixed at 100)
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

    # max possible line-load (for state encoding)
    max_total = GA_profile.max() + GB_profile.max()
    max_val_for_enc = int(np.ceil(max_total))

    Z_t   = np.zeros((n_steps, 7), dtype=int)
    S_tp1 = np.zeros((n_steps, 5), dtype=int)

    # Initial splits for each generator (as %)
    idxA = rng.integers(0, len(L_SPLITS))
    idxB = rng.integers(0, len(L_SPLITS))

    # Convenience: percentage splits (0..1)
    A_L1_pct, A_L2_pct = L_SPLITS[idxA] / 100.0
    B_L2_pct, B_L3_pct = L_SPLITS[idxB] / 100.0

    # Initial generator outputs
    GA0 = GA_profile[0]
    GB0 = GB_profile[0]

    # Initial line loads (actual values, not %):
    PL1 = A_L1_pct * GA0
    PL2 = A_L2_pct * GA0 + B_L2_pct * GB0
    PL3 = B_L3_pct * GB0

    target0 = (GA0 + GB0) / 3.0

    print("Initial loads:",
          PL1, PL2, PL3,
          " total:", PL1 + PL2 + PL3,
          " target per line:", target0)

    actions = [-1, +1]

    for t in range(n_steps):
        GA_t   = GA_profile[t]
        GB_t   = GB_profile[t]
        GA_tp1 = GA_profile[t + 1]
        GB_tp1 = GB_profile[t + 1]

        # current generator states encoded as (P_L1^t, P_L2^t) and (P_L2^t, P_L3^t)
        PL1_i = int(round(PL1))
        PL2_i = int(round(PL2))
        PL3_i = int(round(PL3))

        GA_state = encode_pair(PL1_i, PL2_i, max_val=max_val_for_enc)
        GB_state = encode_pair(PL2_i, PL3_i, max_val=max_val_for_enc)

        # choose AA, AB minimizing unequalness at t+1
        best_cost = np.inf
        best_pairs = []

        # target equal share at t+1
        target_tp1 = (GA_tp1 + GB_tp1) / 3.0

        for aA in actions:
            idxA_cand = max(0, min(len(L_SPLITS) - 1, idxA + aA))            
            A_L1_c_pct, A_L2_c_pct = L_SPLITS[idxA_cand] / 100.0

            for aB in actions:
                idxB_cand = max(0, min(len(L_SPLITS) - 1, idxB + aB))
                B_L2_c_pct, B_L3_c_pct = L_SPLITS[idxB_cand] / 100.0

                # candidate next line loads using *next* generator outputs
                PL1_c = A_L1_c_pct * GA_tp1
                PL2_c = A_L2_c_pct * GA_tp1 + B_L2_c_pct * GB_tp1
                PL3_c = B_L3_c_pct * GB_tp1

                cost = ((PL1_c - target_tp1) ** 2 +
                        (PL2_c - target_tp1) ** 2 +
                        (PL3_c - target_tp1) ** 2)

                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_pairs = [(aA, aB,
                                   idxA_cand, idxB_cand,
                                   PL1_c, PL2_c, PL3_c)]
                elif abs(cost - best_cost) <= 1e-9:
                    best_pairs.append((aA, aB,
                                       idxA_cand, idxB_cand,
                                       PL1_c, PL2_c, PL3_c))

        # break ties randomly if multiple best pairs
        chosen = best_pairs[rng.integers(0, len(best_pairs))]
        AA, AB, idxA_next, idxB_next, PL1_next, PL2_next, PL3_next = chosen

        # store current state + chosen actions, using integer-valued loads
        Z_t[t] = [
            PL1_i,
            PL2_i,
            PL3_i,
            GA_state,
            GB_state,
            AA,
            AB,
        ]

        # update indices for splits
        idxA, idxB = idxA_next, idxB_next

        # next generator states as tuples of new line loads (integers)
        PL1_next_i = int(round(PL1_next))
        PL2_next_i = int(round(PL2_next))
        PL3_next_i = int(round(PL3_next))

        GA_next_state = encode_pair(PL1_next_i, PL2_next_i, max_val=max_val_for_enc)
        GB_next_state = encode_pair(PL2_next_i, PL3_next_i, max_val=max_val_for_enc)

        S_tp1[t] = [
            PL1_next_i,
            PL2_next_i,
            PL3_next_i,
            GA_next_state,
            GB_next_state,
        ]

        # advance continuous loads
        PL1, PL2, PL3 = PL1_next, PL2_next, PL3_next

    return Z_t, S_tp1


def compute_transition_mi_matrix(Z_t, S_tp1):
    """
    Z_t:   (N, 7) array  [P_L1^t, P_L2^t, P_L3^t, S_A^t, S_B^t, A_A^t, A_B^t]
    S_tp1: (N, 5) array  [P_L1^{t+1}, P_L2^{t+1}, P_L3^{t+1}, S_A^{t+1}, S_B^{t+1}]

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


if __name__ == "__main__":
    n_steps = 100_000

    t = np.arange(n_steps + 1)
    GA_profile = discrete_random_state(n_steps + 1, start=100, low=50, high=150,
                                    step_choices=(-5, 0, 5))

    GB_profile = discrete_random_state(n_steps + 1, start=100, low=50, high=150,
                                    step_choices=(-10, -5, 0, 5, 10))

    Z_t, S_tp1 = simulate_transitions(
        n_steps=n_steps,
        GA_profile=GA_profile,
        GB_profile=GB_profile,
    )
    M = compute_transition_mi_matrix(Z_t, S_tp1)

    print("MI matrix shape:", M.shape)
    print(M)
