import numpy as np
import matplotlib.pyplot as plt


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
    Merge all cycles into clusters.
    cycles: list of lists of cluster indices
    """
    merged_sets = []  # final merged clusters
    used = set()

    for cycle in cycles:
        merged = set()
        for idx in cycle:
            used.add(idx)
            merged.update(clusters[idx])
        merged_sets.append(sorted(merged))

    # Add clusters not involved in any cycle
    for i, c in enumerate(clusters):
        if i not in used:
            merged_sets.append(sorted(set(c)))

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


def main():
    '''
    matrix = np.array([[1, 0, 0, 1],
                       [0, 1, 1, 0],
                       [1, 1, 0, 0],
                       [0, 0, 1, 1],
                       [1, 0, 1, 0],
                       [0, 1, 0, 1]])
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
    
    clusters = find_clusters(matrix)
    print("Identified Clusters:")
    for cluster in clusters:
        print(cluster)
    alpha = 0.6
    merged_clusters = merge_clusters(matrix, clusters, alpha)
    print("\nMerged Clusters:")
    for cluster in merged_clusters:
        print(cluster)

if __name__ == "__main__":
    main()
