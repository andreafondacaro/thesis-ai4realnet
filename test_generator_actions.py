import numpy as np
import algorithmFunctions as af
import actionsAlgorithm as aa
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import time


rng = np.random.default_rng()


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

def plot_cluster_graph(matrix, clusters, alpha, p, timetaken=None,
                       curr_matrix_name="Matrix"):
    """
    Graph view with overlapping clusters.
    States and next states are represented by the same node (s0, s1, ...).

    Draws:
      - state -> state edges (only between states in different clusters, no self-loops)
      - action -> state edges (between different clusters as before)
    """
    matrix = np.array(matrix)
    n_states = matrix.shape[1]
    n_actions = matrix.shape[0] - n_states

    # 0..n_states-1                    : states
    # n_states..n_states+n_actions-1   : actions
    # 2*n_states..2*n_states+n_actions-1 : (next states, not used as nodes now)
    combined_size = n_states * 2 + n_actions
    membership = {k: set() for k in range(combined_size)}
    for c_idx, cluster in enumerate(clusters):
        for k in cluster:
            membership[k].add(c_idx)

    # --- Graph and nodes ---
    G = nx.DiGraph()

    state_nodes  = [f"s{i}"  for i in range(n_states)]
    action_nodes = [f"a{j}"  for j in range(n_actions)]

    G.add_nodes_from(state_nodes,  kind="state")
    G.add_nodes_from(action_nodes, kind="action")

    # --- bold states: states that are "controlled" by an action in the same cluster ---
    bold_state_indices = set()
    for cluster in clusters:
        states_in_c  = [k for k in cluster if k < n_states]
        actions_in_c = [k - n_states for k in cluster
                        if n_states <= k < n_states + n_actions]
        for s in states_in_c:
            for a in actions_in_c:
                # fixed: s is already 0..n_states-1, don't check s >= n_states
                if matrix[n_states + a, s] > 0.5:
                    bold_state_indices.add(s)

    bold_state_nodes = {f"s{i}" for i in bold_state_indices}

    # --- Edges ---

    # 1) action -> state edges (bottom n_actions rows)
    for s in range(n_states):
        for a in range(n_actions):
            if matrix[n_states + a, s] < 0.5:
                continue
            state_clusters  = membership[s]
            action_clusters = membership[n_states + a]
            # only if they belong to different clusters
            if state_clusters and action_clusters:
                cross = any(c1 != c2 for c1 in action_clusters for c2 in state_clusters)
            else:
                cross = False
            if cross:
                G.add_edge(f"a{a}", f"s{s}")

    # 2) state -> state edges (top-left n_states x n_states block)
    #    Only between different states, and only if they belong to different clusters.
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue  # no self-loops
            if matrix[i, j] < 0.5:
                continue
            clusters_i = membership[i]
            clusters_j = membership[j]
            if clusters_i and clusters_j:
                cross = any(c1 != c2 for c1 in clusters_i for c2 in clusters_j)
            else:
                cross = False
            if cross:
                G.add_edge(f"s{i}", f"s{j}")

    # --- Layout: cluster-based positions so same-cluster nodes are close ---
    num_clusters = max(1, len(clusters))
    cluster_centers = {}
    radius = 3.0
    for c_idx in range(num_clusters):
        angle = 2 * np.pi * c_idx / num_clusters
        cluster_centers[c_idx] = np.array([radius * np.cos(angle),
                                           radius * np.sin(angle)])

    rng_local = np.random.default_rng(0)  # deterministic jitter
    pos = {}

    def position_for_index(idx, extra_shift=np.array([0.0, 0.0])):
        """idx is in [0, n_states+n_actions); extra_shift used to separate states/actions."""
        cl = membership.get(idx, set())
        if cl:
            centers = np.array([cluster_centers[c] for c in cl])
            base = centers.mean(axis=0)
        else:
            base = np.array([0.0, 0.0])
        jitter = rng_local.normal(scale=0.25, size=2)
        return base + jitter + extra_shift

    # place states and actions
    for i in range(n_states):
        pos[f"s{i}"] = position_for_index(i, extra_shift=np.array([-0.3, 0.0]))
    for j in range(n_actions):
        pos[f"a{j}"] = position_for_index(n_states + j,
                                          extra_shift=np.array([0.3, 0.0]))

    # --- Draw ---
    fig, ax = plt.subplots(figsize=(11, 8))

    # nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=state_nodes,
                           node_shape='s',
                           node_size=700,
                           ax=ax)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=action_nodes,
                           node_shape='o',
                           node_size=700,
                           ax=ax)

    # edges
    nx.draw_networkx_edges(G, pos,
                           arrows=True,
                           arrowstyle='->',
                           arrowsize=18,
                           width=1.5,
                           ax=ax)

    # labels
    labels = {}
    for i in range(n_states):
        labels[f"s{i}"] = f"s{i}"
    for j in range(n_actions):
        labels[f"a{j}"] = f"a{j}"

    normal_nodes = [n for n in G.nodes() if n not in bold_state_nodes]
    nx.draw_networkx_labels(G, pos,
                            labels={n: labels[n] for n in normal_nodes},
                            font_weight='normal',
                            font_size=10,
                            ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels={n: labels[n] for n in bold_state_nodes},
                            font_weight='bold',
                            font_size=10,
                            ax=ax)

    # legend
    state_proxy  = Line2D([0], [0], marker='s', linestyle='None',
                          label='State s', markersize=10)
    action_proxy = Line2D([0], [0], marker='o', linestyle='None',
                          label='Action a', markersize=10)
    bold_proxy   = Line2D([0], [0], marker='s', linestyle='None',
                          label='State controlled in same cluster',
                          markersize=10, markeredgewidth=2)
    ax.legend(handles=[state_proxy, action_proxy, bold_proxy],
              loc='upper right')

    title = f"Cluster Graph (alpha={alpha})"
    if timetaken is not None:
        title += f"  time={timetaken:.2f} ms"
    ax.set_title(title)
    ax.axis('off')

    out_path = rf"C:\Users\andre\Desktop\Uni\Tesi\test_images\{curr_matrix_name}_alpha_{alpha}_p_{p}_graph_{timetaken:.2f}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)



def flip_bits(M, p):
    """
    With prob. p flips each 0→1 and 1→0 independently,
    then enforces:
      1) M[i,i] = 1 for all i in diag
      2) For rows i >= n_cols: each row has at least one 1
      3) For each column: there is at least one 1 in rows i >= n_cols
    """
    M = np.asarray(M)

    flips = rng.random(M.shape) < p
    out = np.where(flips, 1 - M, M).astype(np.uint8)

    n_rows, n_cols = out.shape
    diag_n = min(n_rows, n_cols)

    # 1) Force diagonal to 1
    out[np.arange(diag_n), np.arange(diag_n)] = 1

    # Define "bottom" rows as i >= n_cols (i.e., rows > matrix.shape[1] in your wording)
    start = n_cols
    if start < n_rows:
        bottom = out[start:, :]  # view

        # 2) Each bottom row must have at least one 1
        row_sums = bottom.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        if zero_rows.size > 0:
            cols_pick = rng.integers(0, n_cols, size=zero_rows.size)
            bottom[zero_rows, cols_pick] = 1

        # 3) Each column must have at least one 1 in bottom rows
        col_sums = bottom.sum(axis=0)
        zero_cols = np.where(col_sums == 0)[0]
        if zero_cols.size > 0:
            rows_pick = rng.integers(0, bottom.shape[0], size=zero_cols.size)
            bottom[rows_pick, zero_cols] = 1

    return out



def main():
    
    M1 = np.array([[1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1]])
    
    M2 = np.array([[1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1]])
    
    M3 = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    
    alphas = [0.4, 0.5, 0.6, 0.7]
    probabilities = [0.0, 0.1, 0.3, 0.5]
    matrices = [M1, M2, M3]
    for matrix in matrices:
        if np.array_equal(matrix, M1):
            curr_matrix_name = "M1"
        elif np.array_equal(matrix, M2):
            curr_matrix_name = "M2"
        else:
            curr_matrix_name = "M3"
        for alpha in alphas:
            for p in probabilities:
                print(f"Analyzing matrix with alpha={alpha} and flip probability={p}")
                noisy_matrix = flip_bits(matrix, p)
                new_matrix = np.zeros(noisy_matrix.shape, dtype=float)
                mask0 = (noisy_matrix == 0)
                mask1 = (noisy_matrix == 1)

                new_matrix[mask0] = np.round(rng.uniform(0.01, 0.3, size=mask0.sum()), 3)
                new_matrix[mask1] = np.round(rng.uniform(0.7, 1.0, size=mask1.sum()), 3)
                print(new_matrix)
                start = time.perf_counter()
                clusters = find_clusters(new_matrix)
                merged_clusters = merge_clusters(new_matrix, clusters, alpha)
                for i in range(len(merged_clusters)):
                    merged_clusters[i] = [elem for elem in merged_clusters[i] if elem < new_matrix.shape[0]]
                end = time.perf_counter()
                timetaken = (end - start) * 1000
                print(f"Merged clusters: {merged_clusters}")
                plot_cluster_graph(new_matrix, merged_clusters, alpha=alpha, p=p, timetaken=timetaken, curr_matrix_name=curr_matrix_name)

if __name__ == "__main__":
    main()

