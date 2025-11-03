import numpy as np
import matplotlib.pyplot as plt
import algorithmFunctions as af

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

def plot_original_matrix(matrix, title="Matrix", row_labels=None, col_labels=None):
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

    reordered_matrix, row_order, col_order, clusters = af.analyze_matrix(matrix, alpha=0.6)
    
    plot_original_matrix(matrix, "Matrice Originale")
    af.plot_matrix(reordered_matrix, alpha=0.6, row_labels=row_order, col_labels=col_order, clusters=clusters, curr_matrix_name="Matrice Riordinata per Cluster")

if __name__ == "__main__":
    main()