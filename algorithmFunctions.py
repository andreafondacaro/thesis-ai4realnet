from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt


def evaluate_matrix( matrix, row_indices, col_indices, shape, alpha=0.6):
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

def count_cardinalities( matrix):
    cardinalities = []
    for i in range(matrix.shape[0]):
        cardinality = np.sum(matrix[i, :] == 1)
        cardinalities.append(cardinality)
    for j in range(matrix.shape[1]):
        cardinality = np.sum(matrix[:, j] == 1)
        cardinalities.append(cardinality)
    return cardinalities

def get_index_min( cardinalities, matrix, previous_index):
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


def merge_clusters( matrix, index_min, clusters, alpha):
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

def insert_cluster( matrix, index_min, clusters, alpha):
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
    best_score_index = min(range(len(scores)), key=scores.__getitem__)
    clusters[best_score_index].append(index_min)
    if index_min < matrix.shape[1]:
        clusters[best_score_index].append(index_min + matrix.shape[0])
    elif index_min >= matrix.shape[0]:
        clusters[best_score_index].append(index_min - matrix.shape[0])
    return clusters

def separate_clusters( matrix, cardinalities, alpha):
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

def reorder_matrix_by_clusters( matrix, clusters):
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

def plot_matrix(matrix, alpha, row_labels, col_labels, clusters, p = None,
                timetaken=None, curr_matrix_name="Matrix"):
    """
    Visualizza la matrice binaria (0/1) con celle colorate, numeri e contorni dei cluster.
    I numeri dei cluster si riferiscono all'ordine 'row_order' e 'col_order' (non all'ordine della matrice).
    """
    matrix = np.array(matrix)
    n_rows, n_cols = matrix.shape
    offset = 10  # columns in clusters: offset..offset+n_cols-1
    if n_rows > 10:
        offset = 15

    # fast maps: original index -> position in the PLOTTED matrix
    row_pos = {orig_idx: pos for pos, orig_idx in enumerate(row_labels)}
    col_pos = {orig_idx: pos for pos, orig_idx in enumerate(col_labels)}

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = plt.cm.colors.ListedColormap(['black', 'white'])
    ax.imshow(matrix, cmap=cmap, aspect='equal')
    if p is not None:
        ax.set_title(f"Matrice Riordinata per Cluster (alpha={alpha}, p={p})", fontsize=16)
    else:
        ax.set_title(f"Matrice Riordinata per Cluster (alpha={alpha})", fontsize=16)
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Rows', fontsize=12)

    # numbers in cells
    for i in range(n_rows):
        for j in range(n_cols):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha='center', va='center',
                    color=('white' if value == 0 else 'black'),
                    fontsize=12, fontweight='bold')

    # ticks/labels (assume labels already correspond to the plotted order)
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)), row_labels)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)), col_labels)

    # faint grid
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.tick_params(which='minor', bottom=False, left=False)

    # draw cluster outlines honoring row_order/col_order
    colors = plt.cm.tab10.colors
    legend_handles = []
    for idx, cluster in enumerate(clusters):
        color = colors[idx % len(colors)]
        # decode original indices from cluster encoding
        orig_rows = sorted([r for r in cluster if 0 <= r < offset])
        orig_cols = sorted([c - offset for c in cluster if offset <= c < offset + n_cols])
        # map original indices to positions in the plotted matrix via row_order/col_order
        rows = sorted([row_pos[r] for r in orig_rows if r in row_pos])
        cols = sorted([col_pos[c] for c in orig_cols if c in col_pos])

        if not rows or not cols:
            continue

        # draw 1×1 rectangles on every cross-product cell
        for r in rows:
            for c in cols:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                    fill=False, linewidth=2, edgecolor=color))

        # legend: show indices in the *original* numbering for clarity

        legend_handles.append(
            Line2D([0], [0], color=color, lw=3,
                label=f"Cluster {idx+1}  R:{orig_rows}  C:{orig_cols}")
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    if timetaken is not None:
        out_path = rf"C:\Users\andre\Desktop\Uni\Tesi\test_images\{curr_matrix_name}_alpha_{alpha}_p_{p}_time_{timetaken:.2f}.png"
    else:
        out_path = rf"C:\Users\andre\Desktop\Uni\Tesi\test_images\{curr_matrix_name}_alpha_{alpha}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def analyze_matrix(matrix, alpha):
    cardinalities = count_cardinalities(matrix)
    clusters = separate_clusters(matrix, cardinalities, alpha)
    print(f"Final clusters: {clusters}")
    reordered_matrix, row_order, col_order = reorder_matrix_by_clusters(matrix, clusters)
    return reordered_matrix, row_order, col_order, clusters