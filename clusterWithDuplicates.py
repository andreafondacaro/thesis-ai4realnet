import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import networkx as nx

rng = np.random.default_rng()


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
        

    return separate_score < full_score, full_score

def count_cardinalities(matrix):
    cardinalities = []
    for i in range(matrix.shape[0]):
        cardinality = np.sum(matrix[i, :] == 1)
        cardinalities.append(cardinality)
    for j in range(matrix.shape[1]):
        cardinality = np.sum(matrix[:, j] == 1)
        cardinalities.append(cardinality)
    return cardinalities

def get_index_min(cardinalities, matrix, previous_index):
    min_value = min(val for val in cardinalities if val > 0)
    min_indices = [i for i, val in enumerate(cardinalities) if val == min_value]

    if len(min_indices) == 1:
        index_min = min_indices[0]
    else:
        # Controlla se previous_index e gli indici minimi formano coppie riga-colonna
        for idx in min_indices:
            if previous_index < matrix.shape[0] and idx >= matrix.shape[0]:  # previous riga, idx colonna
                if matrix[previous_index, idx - matrix.shape[0]] == 1:
                    index_min = idx
                    break
            elif previous_index >= matrix.shape[0] and idx < matrix.shape[0]:  # previous colonna, idx riga
                if matrix[idx, previous_index - matrix.shape[0]] == 1:
                    index_min = idx
                    break
        else:
            index_min = min_indices[0]

    return index_min


def merge_clusters(matrix, index_min, clusters, alpha):
    if index_min < matrix.shape[0]:
        indices = np.where(matrix[index_min, :] == 1)[0]
    else:
        indices = np.where(matrix[:, index_min - matrix.shape[0]] == 1)[0]
    merged = True
    while merged:
        merged = False
        for i in range(len(indices)):
            if merged:
                break
            for c1 in range(len(clusters)):
                if merged:
                    break
                if indices[i] in clusters[c1]:
                    for j in range(len(indices)):
                        if merged:
                            break
                        for c2 in range(len(clusters)):
                            if c1 != c2 and indices[j] in clusters[c2]:
                                row_indices = []
                                col_indices = []
                                for k in clusters[c1]:
                                    if k < matrix.shape[0]:
                                        row_indices.append(k)
                                    else:
                                        col_indices.append(k - matrix.shape[0])
                                shape = (len(row_indices), len(col_indices))
                                for k in clusters[c2]:
                                    if k < matrix.shape[0]:
                                        row_indices.append(k)
                                    else:
                                        col_indices.append(k - matrix.shape[0])
                                if not evaluate_matrix(matrix, row_indices, col_indices, shape, alpha):
                                    clusters[c1].extend(clusters[c2])
                                    clusters.pop(c2)
                                    merged = True
                                    break
    
    return clusters

def insert_cluster(matrix, index_min, clusters, alpha):
    print(f"Inserting index {index_min} into clusters {clusters}")
    scores = []
    action_scores = []
    for c in clusters:
        row_indices = []
        col_indices = []
        for k in c:
            if k < matrix.shape[0]:
                row_indices.append(k)
            else:
                col_indices.append(k - matrix.shape[0])
        shape = (len(row_indices), len(col_indices))
        if index_min < matrix.shape[0]:
            row_indices.append(index_min)
            if index_min < matrix.shape[1]:
                col_indices.append(index_min)
        else:
            col_indices.append(index_min - matrix.shape[0])
            row_indices.append(index_min - matrix.shape[0])

        flag, score = evaluate_matrix(matrix, row_indices, col_indices, shape, alpha)
        if index_min < matrix.shape[0] and index_min >= matrix.shape[1]:
            action_scores.append(score)
        if not flag:
            scores.append(score)
        else:
            scores.append(float('inf'))
    if scores == [float('inf')] * len(scores):
        if index_min < matrix.shape[1]:
            clusters.append([index_min, index_min + matrix.shape[0]])
        elif index_min >= matrix.shape[0]:
            clusters.append([index_min - matrix.shape[0], index_min])
        elif action_scores != []:
            best_action_score_index = min(range(len(action_scores)), key=action_scores.__getitem__)
            clusters[best_action_score_index].append(index_min)
        else:
            clusters.append([index_min])
        return clusters
    if index_min >= matrix.shape[1] and index_min < matrix.shape[0]:
        best_score_index = min(range(len(scores)), key=scores.__getitem__)
        clusters[best_score_index].append(index_min)
    else:
        for i in range(len(scores)):
            if scores[i] != float('inf'):
                clusters[i].append(index_min)
                if index_min < matrix.shape[1]:
                    clusters[i].append(index_min + matrix.shape[0])
                elif index_min >= matrix.shape[0]:
                    clusters[i].append(index_min - matrix.shape[0])

    return clusters

def separate_clusters(matrix, cardinalities, alpha):
    cardinalities_original = cardinalities.copy()
    clusters = []
    states_cardinalities = cardinalities[:matrix.shape[1]] + cardinalities[matrix.shape[0]:]
    index_min = min((i for i in range(len(states_cardinalities)) if states_cardinalities[i] > 0), key=states_cardinalities.__getitem__)
    while any(0 < card < float('inf') for card in cardinalities):
        if clusters == []:
            if index_min > matrix.shape[1]:
                index_min = index_min + (len(cardinalities) - 2*matrix.shape[1])
            if index_min < matrix.shape[0]:
                clusters.append([index_min, index_min + matrix.shape[0]])
            else:
                clusters.append([index_min - matrix.shape[0], index_min])
            cardinalities[index_min] = float('inf')
            if index_min < matrix.shape[0]:
                cardinalities[index_min + matrix.shape[0]] = float('inf')
            else:
                cardinalities[index_min - matrix.shape[0]] = float('inf')
            if index_min < matrix.shape[0]:
                for col in range(matrix.shape[1]):
                    if matrix[index_min, col] == 1 and cardinalities[col] != float('inf'):
                        cardinalities[col] -= 1   
            else:
                for row in range(matrix.shape[0]):
                    if matrix[row, index_min - matrix.shape[0]] == 1 and cardinalities[row] != float('inf'):
                        cardinalities[row] -= 1
            index_min = get_index_min(cardinalities, matrix, index_min)
            continue
        if cardinalities_original[index_min] >= 2:
            #checks if 2 clusters are now worth merging
            clusters = merge_clusters(matrix, index_min, clusters, alpha)
        #find the best score cluster and add into it
        clusters = insert_cluster(matrix, index_min, clusters, alpha)

        cardinalities[index_min] = float('inf')
        if index_min < matrix.shape[1]:
            cardinalities[index_min + matrix.shape[0]] = float('inf')
        elif index_min >= matrix.shape[0]:
            cardinalities[index_min - matrix.shape[0]] = float('inf')
        if index_min < matrix.shape[0]:
            for col in range(matrix.shape[1]):
                if matrix[index_min, col] == 1 and cardinalities[col] != float('inf'):
                    cardinalities[col] -= 1   
        else:
            for row in range(matrix.shape[0]):
                if matrix[row, index_min - matrix.shape[0]] == 1 and cardinalities[row] != float('inf'):
                    cardinalities[row] -= 1
        index_min = get_index_min(cardinalities, matrix, index_min)


    return clusters

def reorder_matrix_by_clusters(matrix, clusters):
    n_rows = matrix.shape[0]
    
    row_order = []
    col_order = []
    
    for cluster in clusters:
        cluster_rows = [idx for idx in cluster if idx < n_rows]
        cluster_cols = [idx - n_rows for idx in cluster if idx >= n_rows]
        row_order.extend(cluster_rows)
        col_order.extend(cluster_cols)
    
    reordered_matrix = matrix[np.ix_(row_order, col_order)]
    
    return reordered_matrix, row_order, col_order

import networkx as nx

import networkx as nx

def plot_cluster_graph(matrix, clusters, alpha, p, timetaken=None,
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
    n_states, n_actions = matrix.shape

    # --- Cluster membership in "combined" index space (as in your code) ---
    # 0..n_states-1                : states
    # n_states..n_states+n_actions-1 : actions
    combined_size = n_states + n_actions
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

    # --- Bold states: controlled by actions in the same cluster ---
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

    # --- Edges from actions -> states (cross-cluster only) ---
    for s in range(n_states):
        for a in range(n_actions):
            if matrix[s, a] != 1:
                continue
            state_clusters  = membership[s]
            action_clusters = membership[n_states + a]
            cross = any(c1 != c2 for c1 in action_clusters for c2 in state_clusters)
            if cross:
                G.add_edge(f"a{a}", f"s{s}")   # <-- ONLY actions as sources

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

    title = f"Cluster Graph (alpha={alpha}, p={p})"
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
    With prob. p flips each 0→1 and 1→0 independently.
    """
    flips = rng.random(M.shape) < p
    return np.where(flips, 1-M, M)

def analyze_matrix(matrix, alpha):
    cardinalities = count_cardinalities(matrix)
    print(f"cardinalities: {cardinalities}")
    clusters = separate_clusters(matrix, cardinalities, alpha)
    print(f"Final clusters: {clusters}")
    reordered_matrix, row_order, col_order = reorder_matrix_by_clusters(matrix, clusters)
    return reordered_matrix, row_order, col_order, clusters

def main():
    
    #M2 has to be 10x5 diagonal with 4 clusters
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
    
    alphas = [0.4, 0.5, 0.6, 0.7]
    probabilities = [0.0, 0.1, 0.3, 0.5]
    matrices = [M1, M2]
    '''
    for matrix in matrices:
        if np.array_equal(matrix, M1):
            curr_matrix_name = "M1"
        else:
            curr_matrix_name = "M2"
        for alpha in alphas:
            for p in probabilities:
                print(f"Analyzing matrix with alpha={alpha} and flip probability={p}")
                noisy_matrix = flip_bits(matrix, p)
                start = time.perf_counter()
                reordered_matrix, row_order, col_order, clusters = analyze_matrix(noisy_matrix, alpha)
                end = time.perf_counter()
                plot_matrix(reordered_matrix, alpha, p, clusters, timetaken=(end-start)*1000, row_labels=row_order, col_labels=col_order, curr_matrix_name=curr_matrix_name)
'''
    print(f"Analyzing matrix with alpha={0.4} and flip probability={0.3}")
    noisy_matrix = flip_bits(M1, 0.1)
    start = time.perf_counter()

    reordered_matrix, row_order, col_order, clusters = analyze_matrix(noisy_matrix, 0.4)
    end = time.perf_counter()
    print(f"Reordered matrix:\n{noisy_matrix}")
    #plot_matrix(reordered_matrix, 0.4, 0.3, clusters, timetaken=(end-start)*1000, row_labels=row_order, col_labels=col_order, curr_matrix_name="M1")
    plot_cluster_graph(noisy_matrix, clusters, 0.4, 0.1, timetaken=(end-start)*1000, curr_matrix_name="M1")

if __name__ == "__main__":
    main()

