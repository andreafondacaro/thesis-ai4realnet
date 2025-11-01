import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(blocks_diagonal):
    if blocks_diagonal:
        return create_block_diagonal_matrix()
    else:
        rows = np.random.randint(5, 15)
        cols = rows - np.random.randint(2, rows - 2)
        return np.random.choice([0, 1], size=(rows, cols), p=[0.7, 0.3])

def create_block_diagonal_matrix():
    num_blocks = np.random.randint(2, 5)
    blocks = []
    
    for _ in range(num_blocks):
        rows = np.random.randint(2, 6)
        cols = np.random.randint(2, 6)
        block = np.ones((rows, cols), dtype=int)
        blocks.append(block)
    
    return blocks_to_diagonal(blocks)

def blocks_to_diagonal(blocks):
    total_rows = sum(block.shape[0] for block in blocks)
    total_cols = sum(block.shape[1] for block in blocks)
    
    matrix = np.zeros((total_rows, total_cols), dtype=int)
    
    row_offset = 0
    col_offset = 0
    
    for block in blocks:
        rows, cols = block.shape
        matrix[row_offset:row_offset+rows, col_offset:col_offset+cols] = block
        row_offset += rows
        col_offset += cols
    
    return matrix

def evaluate_matrix(matrix, row_indices, col_indices, shape):
    #computes scores for the 2 separate matrices and compares them to the full matrix
    full_score = 0
    separate_score = 0
    alpha = 0.6
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


def merge_clusters(matrix, index_min, clusters):
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
                                if not evaluate_matrix(matrix, row_indices, col_indices, shape):
                                    clusters[c1].extend(clusters[c2])
                                    clusters.pop(c2)
                                    merged = True
                                    break
    
    print(f"Clusters after merging: {clusters}")
    return clusters

def insert_cluster(matrix, index_min, clusters):
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
            print("index_min:", index_min)
            print(index_min - matrix.shape[0])
            row_indices.append(index_min - matrix.shape[0])

        flag, score = evaluate_matrix(matrix, row_indices, col_indices, shape)
        if index_min < matrix.shape[0] and index_min >= matrix.shape[1]:
            action_scores.append(score)
        if not flag:
            scores.append(score)
        else:
            scores.append(float('inf'))
    print(f"Scores: {scores}")
    if scores == [float('inf')] * len(scores):
        if index_min < matrix.shape[1]:
            clusters.append([index_min, index_min + matrix.shape[0]])
        elif index_min >= matrix.shape[0]:
            clusters.append([index_min - matrix.shape[0], index_min])
        elif action_scores != []:
            print("Action scores:", action_scores)
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

def separate_clusters(matrix, cardinalities):
    cardinalities_original = cardinalities.copy()
    clusters = []
    states_cardinalities = cardinalities[:matrix.shape[1]] + cardinalities[matrix.shape[0]:]
    print(f"States cardinalities: {states_cardinalities}")
    index_min = min((i for i in range(len(states_cardinalities)) if states_cardinalities[i] > 0), key=states_cardinalities.__getitem__)
    while any(0 < card < float('inf') for card in cardinalities):
        print("index_min:", index_min)
        if clusters == []:
            if index_min > matrix.shape[1]:
                index_min = index_min + (len(cardinalities) - 2*matrix.shape[1])
                print("index_min adjusted:", index_min)
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
            clusters = merge_clusters(matrix, index_min, clusters)
        #find the best score cluster and add into it
        clusters = insert_cluster(matrix, index_min, clusters)

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
        print(clusters)
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

def plot_matrix(matrix, title="Matrix", row_labels=None, col_labels=None):
    """Visualizza la matrice binaria (0/1) con celle colorate e numeri."""
    matrix = np.array(matrix)
    
    plt.figure(figsize=(12, 10))
    
    cmap = plt.cm.colors.ListedColormap(['black', 'white'])
    
    im = plt.imshow(matrix, cmap=cmap, aspect='equal')

    plt.title(title, fontsize=16)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Rows', fontsize=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            text_color = 'white' if value == 0 else 'black'
            plt.text(j, i, str(value), ha='center', va='center',
                     color=text_color, fontsize=12, fontweight='bold')

    if row_labels is not None:
        plt.yticks(range(len(row_labels)), row_labels)
    if col_labels is not None:
        plt.xticks(range(len(col_labels)), col_labels)
        
    plt.show()



def main():
    block_diagonal = True
    #matrix = generate_matrix(block_diagonal)
    
    matrix = np.array([
        [1,1,1,0,0,0],
        [1,1,1,0,0,0],
        [0,1,1,1,0,0],
        [0,0,1,1,1,0],
        [0,0,0,1,1,1],
        [0,0,0,0,1,1],
        [1,1,0,0,0,1],
        [1,1,1,0,0,0],
        [0,0,1,1,1,0],
        [0,0,0,1,1,1],
        [1,1,0,1,0,0],
        [0,0,1,1,0,1],
        [1,1,1,0,1,0],
    ], dtype=int)

    '''
    rng = np.random.default_rng(42)
    matrix = np.zeros((20, 8), dtype=int)

    # base clusters (broad overlapping ones)
    base_patterns = [
        [1,1,1,0,0,0,0,0],
        [0,1,1,1,0,0,0,0],
        [0,0,1,1,1,0,0,0],
        [0,0,0,1,1,1,0,0],
        [0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,1,1],
        [1,0,0,0,0,1,0,1],
        [1,1,0,0,0,0,1,0]
    ]

    # fill rows with pattern + some noise
    for i in range(20):
        base = np.array(base_patterns[i % len(base_patterns)])
        # add diagonal 1
        if i < 8:
            base[i] = 1
        # add random overlaps and flips
        extras = rng.choice(8, size=rng.integers(0,2), replace=False)
        base[extras] = 1
        flips = rng.choice(8, size=rng.integers(0,2), replace=False)
        for f in flips:
            base[f] = 1 - base[f]
        matrix[i] = base
    # enforce diagonal 1s
    for i in range(min(matrix.shape)):
        matrix[i, i] = 1
    matrix = np.array([
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [0,0,0,0,1]])

    matrix = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0]])
    
    matrix = np.array([[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],    
       [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
'''

    print(matrix)
    cardinalities = count_cardinalities(matrix)
    print(f"Cardinalities: {cardinalities}")

    clusters = separate_clusters(matrix, cardinalities)
    print(f"Clusters: {clusters}")

    reordered_matrix, row_order, col_order = reorder_matrix_by_clusters(matrix, clusters)
    
    plot_matrix(matrix, "Matrice Originale")
    plot_matrix(reordered_matrix, "Matrice Riordinata per Cluster", row_labels=row_order, col_labels=col_order)

if __name__ == "__main__":
    main()

    #TODO DEVE PRENDERE IL MINIMO COLLEGATO AL NODO PRECEDENTE, SALVA IL NODO PRECEDENTE IN VARIABILE PREVIOUS INDEX E POI OGNI VOLTA CHE SCEGLI IL MINIMO SE CE NE SONO PIù UGUALI DEVE SCEGLIERE QUELLO COLLEGATO AL NODO PRECEDENTE