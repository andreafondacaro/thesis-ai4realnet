import numpy as np
from numpy.random import default_rng
import actionsAlgorithm

rng = default_rng(0)

def make_factored_binary_matrix(
    n_states: int,
    n_actions: int,
    cluster_sizes=None,
    n_clusters: int = None,     
    intra_prob: float = 1.0,
    inter_prob: float = 0.05,
):
    """
    Create a (n_states + n_actions) x n_states 0/1 matrix M with:
      • Rows 0..n_states-1 = current states
      • Rows n_states..n_states+n_actions-1 = actions
      • Columns 0..n_states-1 = next states

    Constraints:
      1) For each state row i, M[i, i] = 1.
      2) Each column j has at least one 1 in some action row.

    Structure:
      • States are partitioned into factors with given cluster_sizes.
      • Intra-factor connections are dense-ish (prob = intra_prob).
      • Inter-factor connections are sparse (prob = inter_prob).

    Returns:
      M            : np.ndarray, shape ((n_states + n_actions), n_states), dtype=int
      state_label  : np.ndarray, shape (n_states,)   factor id for each state dim
      action_label : np.ndarray, shape (n_actions,)  factor id for each action row
    """

    # --- Handle cluster_sizes / n_clusters ---
    if cluster_sizes is not None:
        cluster_sizes = list(cluster_sizes)
        n_clusters = len(cluster_sizes)
    else:
        block_size = n_states // n_clusters
        cluster_sizes = [block_size] * n_clusters

    n_clusters = len(cluster_sizes)
    n_rows = n_states + n_actions

    state_label = np.concatenate([
        np.full(sz, f, dtype=int) for f, sz in enumerate(cluster_sizes)
    ])

    boundaries = np.concatenate(([0], np.cumsum(cluster_sizes)))

    # Assign actions to clusters proportional to cluster size
    cluster_sizes_arr = np.array(cluster_sizes, dtype=float)
    raw_quotas = cluster_sizes_arr * (n_actions / float(n_states))
    base = np.floor(raw_quotas).astype(int)
    remaining = n_actions - base.sum()
    frac = raw_quotas - base
    order = np.argsort(-frac)
    for f in order[:remaining]:
        base[f] += 1

    # Build action_label: repeat cluster index according to base[f]
    action_label = []
    for f, count in enumerate(base):
        action_label.extend([f] * count)
    action_label = np.array(action_label, dtype=int)
    assert len(action_label) == n_actions

    M = np.zeros((n_rows, n_states), dtype=int)

    #STATE rows: intra- & inter-cluster edges + forced diagonal ones
    for f in range(n_clusters):
        r_start = boundaries[f]
        r_end   = boundaries[f+1]
        sz_f = r_end - r_start

        # Intra-cluster (rows in cluster f, cols in cluster f)
        intra_mask = rng.random((sz_f, sz_f)) < intra_prob
        M[r_start:r_end, r_start:r_end] = intra_mask.astype(int)

        # Cross-cluster links
        for g in range(n_clusters):
            if g == f:
                continue
            c_start = boundaries[g]
            c_end   = boundaries[g+1]
            sz_g = c_end - c_start
            cross_mask = rng.random((sz_f, sz_g)) < inter_prob
            M[r_start:r_end, c_start:c_end] = np.maximum(
                M[r_start:r_end, c_start:c_end],
                cross_mask.astype(int),
            )

    # Enforce diagonal ones
    for i in range(n_states):
        M[i, i] = 1

    #ACTION rows: same thing as state rows
    for a in range(n_actions):
        f = action_label[a]
        r = n_states + a

        c_start = boundaries[f]
        c_end   = boundaries[f+1]
        sz_f = c_end - c_start
        intra_cols = rng.random(sz_f) < intra_prob
        M[r, c_start:c_end] = np.maximum(M[r, c_start:c_end], intra_cols.astype(int))

        for g in range(n_clusters):
            if g == f:
                continue
            cs = boundaries[g]
            ce = boundaries[g+1]
            sz_g = ce - cs
            cross_cols = rng.random(sz_g) < inter_prob
            M[r, cs:ce] = np.maximum(M[r, cs:ce], cross_cols.astype(int))

    #Ensure each column has at least one action '1'
    action_rows = np.arange(n_states, n_rows)
    for j in range(n_states):
        if M[action_rows, j].sum() == 0:
            f = state_label[j]
            candidate_actions = np.where(action_label == f)[0]
            if len(candidate_actions) == 0:
                candidate_actions = np.arange(n_actions)
            a_choice = int(rng.choice(candidate_actions))
            M[n_states + a_choice, j] = 1

    return M, state_label, action_label




# ---------- Run the full pipeline ----------
if __name__ == "__main__":
    n_s, n_a, k = 12, 12, 3
    M, state_label, action_label = make_factored_binary_matrix(n_states=n_s, n_actions=n_a, n_clusters=k,
                                                      intra_prob=0.8, inter_prob=0.2)
    print("Generated Matrix M:")
    print(M)
    clusters = actionsAlgorithm.cluster_matrix(M, alpha=0.6)
