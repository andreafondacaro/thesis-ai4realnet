import numpy as np
import algorithmFunctions as af

import time


rng = np.random.default_rng()


def flip_bits(M, p):
    """
    With prob. p flips each 0→1 and 1→0 independently.
    """
    flips = rng.random(M.shape) < p
    np.fill_diagonal(flips, False)
    return np.where(flips, 1-M, M)



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
    
    alphas = [0.4, 0.5, 0.6, 0.7]
    probabilities = [0.0, 0.1, 0.3, 0.5]
    matrices = [M1, M2]
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
                reordered_matrix, row_order, col_order, clusters = af.analyze_matrix(noisy_matrix, alpha)
                end = time.perf_counter()
                af.plot_matrix(reordered_matrix, alpha=alpha, p=p, clusters=clusters, timetaken=(end-start)*1000, row_labels=row_order, col_labels=col_order, curr_matrix_name=curr_matrix_name)

if __name__ == "__main__":
    main()

