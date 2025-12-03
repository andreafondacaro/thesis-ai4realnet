import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import time
rng = np.random.default_rng(0) 


def find_clusters(matrix):
    clusters = []
    rows, cols = matrix.shape
    for i in range(cols, rows):
        curr_cluster = [i]
        for j in range(cols):
            if matrix[i, j] == 1:
                curr_cluster.append(j)
                curr_cluster.append(j + rows)
        clusters.append((curr_cluster))
    return clusters

def influence_fn(matrix, state, action):
    """
    Determines if action influences state based on matrix.

    An action influences a state if matrix[state, action] == 1.
    """
    return matrix[action, state - matrix.shape[0]] == 1

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

def evaluate_matrix(matrix, row_indices, col_indices, shape, alpha=0.6):
#computes scores for the 2 separate matrices and compares them to the full matrix
    full_score = 0
    separate_score = 0
    full_matrix = matrix[np.ix_(row_indices, col_indices)]

    for i in range(full_matrix.shape[0]):
        for j in range(full_matrix.shape[1]):
            if full_matrix[i, j] == 0:
                full_score += alpha * 1
            if i < shape[0] and j < shape[1] and full_matrix[i, j] == 0:
                separate_score += alpha * 1
            elif i >= shape[0] and j >= shape[1] and full_matrix[i, j] == 0:
                separate_score += alpha * 1
            elif (i < shape[0] and j >= shape[1]) or (i >= shape[0] and j < shape[1]):
                if full_matrix[i, j] == 1:
                    separate_score += (1 - alpha) * 1
    print(f"Full score: {full_score}, Separate score: {separate_score}")
        

    return separate_score > full_score, full_score

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

def merge_cycles(clusters, cycles):
    """
    Merge clusters according to cycles.

    - `clusters`: list of lists of elements (e.g. variable indices)
    - `cycles`  : list of lists of *cluster indices* (e.g. [0,2,3])

    All clusters that appear in the same connected component of the
    "cycle graph" are merged into a single cluster.

    Returns: list of sorted lists, with no duplicates.
    """
    n = len(clusters)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1) Use cycles to union cluster indices
    used_indices = set()
    for cycle in cycles:
        if not cycle:
            continue
        used_indices.update(cycle)
        base = cycle[0]
        for idx in cycle[1:]:
            union(base, idx)

    # 2) Build groups: root -> union of all elements in those clusters
    groups = {}
    for i, c in enumerate(clusters):
        root = find(i) if i in used_indices else i   # untouched clusters stay alone
        groups.setdefault(root, set()).update(c)

    # 3) Return list of unique merged clusters (no duplicates)
    merged_sets = [sorted(s) for s in groups.values()]
    return merged_sets


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
        clusters = merge_cycles(clusters, cycles)
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
                merged = sorted(set(cluster_a) | set(cluster_b))

                # 2) compute row/col indices from merged cluster
                row_indices = [idx for idx in merged if idx < matrix.shape[0]]
                col_indices = [idx - matrix.shape[0] for idx in merged if idx >= matrix.shape[0]]

                cluster_sizes = (len(cluster_a) // 2, len(cluster_b) // 2)

                can_merge, score = evaluate_matrix(matrix, row_indices, col_indices, cluster_sizes, alpha)

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

def plot_cluster_graph(matrix, clusters, alpha, timetaken=None,
                       curr_matrix_name="Matrix"):
    """
    Graph view with overlapping clusters.
    """
    matrix = np.array(matrix)
    n_states = matrix.shape[1]
    n_actions = matrix.shape[0] - n_states

    # 0..n_states-1                : states
    # n_states..n_states+n_actions-1 : actions
    combined_size = n_states*2 + n_actions
    membership = {k: set() for k in range(combined_size)}
    for c_idx, cluster in enumerate(clusters):
        for k in cluster:
            membership[k].add(c_idx)

    # --- Graph and nodes ---
    G = nx.DiGraph()

    state_nodes = [f"s{i}"   for i in range(n_states)]
    action_nodes = [f"a{j}"  for j in range(n_actions)]
    next_nodes   = [f"sp{i}" for i in range(n_states)]

    #G.add_nodes_from(state_nodes, kind="state")
    G.add_nodes_from(action_nodes, kind="action")
    G.add_nodes_from(next_nodes,   kind="next_state")

    bold_state_indices = set()
    for cluster in clusters:
        states_in_c  = [k for k in cluster if k < n_states]
        actions_in_c = [k - n_states for k in cluster
                        if n_states <= k < n_states + n_actions]
        for s in states_in_c:
            for a in actions_in_c:
                if matrix[n_states + a, s] == 1 and s >= n_states:
                    bold_state_indices.add(s)

    bold_state_nodes = {f"s{i}" for i in bold_state_indices}

    # edges creation only action next state
    for s in range(n_states):
        for a in range(n_actions):
            if matrix[n_states + a, s] != 1:
                continue
            state_clusters  = membership[s]
            action_clusters = membership[n_states + a]
            cross = any(c1 != c2 for c1 in action_clusters for c2 in state_clusters)
            if cross:
                G.add_edge(f"a{a}", f"sp{s}")

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
        """idx is in [0, n_states+n_actions); extra_shift used for s' nodes."""
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

    # place next states near their corresponding states
    for i in range(n_states):
        pos[f"sp{i}"] = pos[f"s{i}"] + np.array([0.0, 0.5])  # small offset above

    # --- Draw ---
    fig, ax = plt.subplots(figsize=(11, 8))

    # nodes
    '''nx.draw_networkx_nodes(G, pos,
                           nodelist=state_nodes,
                           node_shape='s',
                           node_size=700,
                           ax=ax)'''
    nx.draw_networkx_nodes(G, pos,
                           nodelist=action_nodes,
                           node_shape='o',
                           node_size=700,
                           ax=ax)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=next_nodes,
                           node_shape='s',
                           node_size=500,
                           ax=ax)

    # edges (only from actions)
    nx.draw_networkx_edges(G, pos,
                           arrows=True,
                           arrowstyle='->',
                           arrowsize=18,
                           width=1.5,
                           ax=ax)

    # labels:
    labels = {}
    for i in range(n_states):
        labels[f"s{i}"] = f"s{i}"
        labels[f"sp{i}"] = f"s'{i}"
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
    state_proxy = Line2D([0], [0], marker='s', linestyle='None',
                         label='State s', markersize=10)
    next_proxy  = Line2D([0], [0], marker='s', linestyle='None',
                         label="Next state s'", markersize=8)
    action_proxy = Line2D([0], [0], marker='o', linestyle='None',
                          label='Action a', markersize=10)
    bold_proxy  = Line2D([0], [0], marker='s', linestyle='None',
                         label='State controlled in same cluster',
                         markersize=10, markeredgewidth=2)
    ax.legend(handles=[state_proxy, next_proxy, action_proxy, bold_proxy],
              loc='upper right')

    title = f"Cluster Graph (alpha={alpha})"
    if timetaken is not None:
        title += f"  time={timetaken:.2f} ms"
    ax.set_title(title)
    ax.axis('off')

    out_path = rf"C:\Users\andre\Desktop\Uni\Tesi\test_images\{curr_matrix_name}_alpha_{alpha}_graph_{timetaken:.2f}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def flip_bits(M, p):
    """
    With prob. p flips each 0→1 and 1→0 independently.
    """
    flips = rng.random(M.shape) < p
    return np.where(flips, 1-M, M)

def main():
    '''
    matrix = np.array([[1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1]])
    '''
    matrix = np.array([[1, 1, 0, 0, 0],
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
    
    matrix = flip_bits(matrix, 0.0)
    start = time.perf_counter()
    clusters = find_clusters(matrix)
    print("Identified Clusters:")
    for cluster in clusters:
        print(cluster)
    alpha = 0.6
    merged_clusters = merge_clusters(matrix, clusters, alpha)
    end = time.perf_counter()
    timetaken = (end - start) * 1000
    print("\nMerged Clusters:")
    for cluster in merged_clusters:
        print(cluster)
    plot_cluster_graph(matrix, merged_clusters, alpha=0.6, timetaken=timetaken)

if __name__ == "__main__":
    main()
