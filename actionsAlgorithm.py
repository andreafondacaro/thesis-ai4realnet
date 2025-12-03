import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import time
rng = np.random.default_rng(0)  # for reproducibility



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
        

    return separate_score > full_score, full_score

def merge_clusters(matrix, clusters, alpha):
    """
    Repeatedly evaluates possible merges and allows clusters to merge multiple times.
    Ensures each index appears only once within and across clusters.
    Duplicate resolution strategy: keep-first (later clusters drop duplicates).
    """

    def unique_preserve_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    changed = True
    pass_num = 0

    # Pre-clean individual clusters
    clusters = [unique_preserve_order(cl) for cl in clusters]

    while changed:
        pass_num += 1
        changed = False
        new_clusters = []
        used = set()

        for i in range(len(clusters)):
            if i in used:
                continue

            current = unique_preserve_order(clusters[i])
            used.add(i)

            while True:
                best_j = None
                best_score = float('inf')
                best_merge = None

                for j in range(len(clusters)):
                    if j == i or j in used:
                        continue

                    candidate = unique_preserve_order(clusters[j])

                    # Tentative merge (dedup within the merged cluster)
                    merged = unique_preserve_order(current + candidate)

                    # Build row/col indices for evaluation
                    row_indices = [idx for idx in merged if idx < matrix.shape[0]]
                    col_indices = [idx - matrix.shape[0] for idx in merged if idx >= matrix.shape[0]]

                    can_merge, score = evaluate_matrix(
                        matrix,
                        row_indices,
                        col_indices,
                        (len(current)//2, len(candidate)//2),
                        alpha
                    )
                    if can_merge and score < best_score:
                        best_score = score
                        best_merge = merged
                        best_j = j

                if best_j is not None and best_merge is not None:
                    current = best_merge
                    used.add(best_j)
                    changed = True
                else:
                    break

            new_clusters.append(current)
        print(f"\nAfter pass {pass_num}, merged clusters:")
        print(new_clusters)
        untouched = [clusters[k] for k in range(len(clusters)) if k not in used]
        if untouched:
            # Any untouched clusters are appended then globally resolved
            combined = new_clusters + [unique_preserve_order(cl) for cl in untouched]
            #combined = resolve_global_uniqueness(combined)
        else:
            combined = new_clusters

        clusters = combined

    print(f"\nFinal merged clusters: {clusters}")
    return clusters


def cluster_matrix(matrix, alpha):
    clusters = find_clusters(matrix)
    print("Identified Clusters:")
    for cluster in clusters:
        print(cluster)
    merged_clusters = merge_clusters(matrix, clusters, alpha)
    return merged_clusters

def plot_cluster_graph(matrix, clusters, alpha, timetaken=None,
                       curr_matrix_name="Matrix"):
    """
    Graph view with overlapping clusters.

    - Rows  : state variables s0, s1, ...
    - Cols  : action variables a0, a1, ...
    - Extra: next-state nodes s'0, s'1, ... (one per state) just for
      visualization / clustering.
    - Bold states: states that are controlled by at least one action in
      the *same* cluster (matrix[row, col] == 1 and both belong to that cluster).
    - Edges: ONLY from actions to states, and only when the action / state
      belong to different clusters (cross-cluster control).
    - Layout: nodes are placed near the centers of the clusters they belong to,
      so variables in the same cluster tend to be close.
    """
    matrix = np.array(matrix)
    n_states = matrix.shape[1]
    n_actions = matrix.shape[0] - n_states

    # --- Cluster membership in "combined" index space (as in your code) ---
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
    next_nodes   = [f"sp{i}" for i in range(n_states)]   # internal name for s'

    G.add_nodes_from(state_nodes, kind="state")
    G.add_nodes_from(action_nodes, kind="action")
    G.add_nodes_from(next_nodes,   kind="next_state")

    bold_state_indices = set()
    for cluster in clusters:
        states_in_c  = [k for k in cluster if k < n_states]
        actions_in_c = [k - n_states for k in cluster
                        if n_states <= k < n_states + n_actions]
        for s in states_in_c:
            for a in actions_in_c:
                if matrix[s, a] == 1:
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
                G.add_edge(f"a{a}", f"sp{s}")   # <-- ONLY actions as sources

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
              loc='upper left')

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
    
    matrix = flip_bits(matrix, p=0.3)  # Introduce some noise
    print(matrix)
    start = time.perf_counter()
    merged_clusters = cluster_matrix(matrix, alpha=0.6)
    end = time.perf_counter()
    timetaken = (end - start) * 1000  # Convert to milliseconds
    print("Merged Clusters:")
    for cluster in merged_clusters:
        print(cluster)
    plot_cluster_graph(matrix, merged_clusters, alpha=0.6, timetaken=timetaken)

if __name__ == "__main__":
    main()
