import numpy as np

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
])  # discrete states along which we move adjacently


def encode_pair(a, b, max_val=200):
    """Encode a pair (a, b) into a single integer for mi_discrete."""
    return a * (max_val + 1) + b


def simulate_transitions(n_steps=100_000):
    """
    Z_t[t]   = (P_L1^t, P_L2^t, P_L3^t,
                G_A^t, G_B^t, A_A^t, A_B^t)       shape (N, 7)

    S_tp1[t] = (P_L1^{t+1}, P_L2^{t+1}, P_L3^{t+1},
                G_A^{t+1}, G_B^{t+1})             shape (N, 5)

    Total load is always 200:
      - Generator A outputs 100 split across L1 and L2.
      - Generator B outputs 100 split across L2 and L3.

    Generators are always "on" and their states are:
      G_A^t = (P_L1^t, P_L2^t)
      G_B^t = (P_L2^t, P_L3^t)

    At each step, actions A_A, A_B ∈ {-1, 0, +1} are chosen to move the
    generators' splits only to adjacent states in L_SPLITS, with the
    control objective of making (P_L1, P_L2, P_L3) as equal as possible.
    """

    Z_t   = np.zeros((n_steps, 7), dtype=int)
    S_tp1 = np.zeros((n_steps, 5), dtype=int)

    target = 200.0 / 3.0
    actions = [-1, +1]

    # Initial splits for each generator
    idxA = rng.integers(0, len(L_SPLITS))
    idxB = rng.integers(0, len(L_SPLITS))

    # Generator A: (A_L1, A_L2)
    A_L1, A_L2 = L_SPLITS[idxA]
    # Generator B: (B_L2, B_L3)
    B_L2, B_L3 = L_SPLITS[idxB]

    # Initial line loads (total = 200)
    PL1 = A_L1
    PL2 = A_L2 + B_L2
    PL3 = B_L3

    print("Initial loads:", PL1, PL2, PL3,
          " total:", PL1 + PL2 + PL3)

    for t in range(n_steps):
        # current generator states as tuples of line loads
        GA = encode_pair(PL1, PL2)
        GB = encode_pair(PL2, PL3)

        # we will choose AA, AB by minimizing the "unequalness" cost
        best_cost = np.inf
        best_pairs = []

        for aA in actions:
            idxA_cand = (idxA + aA) % len(L_SPLITS)
            A_L1_c, A_L2_c = L_SPLITS[idxA_cand]

            for aB in actions:
                idxB_cand = (idxB + aB) % len(L_SPLITS)
                B_L2_c, B_L3_c = L_SPLITS[idxB_cand]

                # candidate next line loads
                PL1_c = A_L1_c
                PL2_c = A_L2_c + B_L2_c
                PL3_c = B_L3_c

                # cost: how far from equal sharing
                cost = ((PL1_c - target) ** 2 +
                        (PL2_c - target) ** 2 +
                        (PL3_c - target) ** 2)

                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_pairs = [(aA, aB, idxA_cand, idxB_cand,
                                   PL1_c, PL2_c, PL3_c)]
                elif abs(cost - best_cost) <= 1e-9:
                    best_pairs.append((aA, aB, idxA_cand, idxB_cand,
                                       PL1_c, PL2_c, PL3_c))

        # break ties randomly if multiple best pairs
        chosen = best_pairs[rng.integers(0, len(best_pairs))]
        AA, AB, idxA_next, idxB_next, PL1_next, PL2_next, PL3_next = chosen

        # store current state + chosen actions
        Z_t[t] = [PL1, PL2, PL3, GA, GB, AA, AB]

        # update indices
        idxA, idxB = idxA_next, idxB_next

        # next generator states as tuples of new line loads
        GA_next = encode_pair(PL1_next, PL2_next)
        GB_next = encode_pair(PL2_next, PL3_next)

        # store next state
        S_tp1[t] = [PL1_next, PL2_next, PL3_next, GA_next, GB_next]

        # advance
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
    Z_t, S_tp1 = simulate_transitions(n_steps=100_000)
    M = compute_transition_mi_matrix(Z_t, S_tp1)
    #M = (M > 0.3).astype(int)

    print("MI matrix shape:", M.shape)   # (7, 5)
    print(M)