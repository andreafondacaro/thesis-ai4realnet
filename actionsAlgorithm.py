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

def merge_clusters(matrix, clusters, alpha, max_iter=None):
    """
    Greedy merging of clusters:
    - Always merge the single best pair (lowest score) if evaluate_matrix says it's beneficial.
    - Repeat until no beneficial merge exists (or max_iter is hit).
    - Merged clusters have duplicates removed.
    """
    clusters = [list(set(c)) for c in clusters]

    it = 0
    while True:
        it += 1
        if max_iter is not None and it > max_iter:
            break

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
