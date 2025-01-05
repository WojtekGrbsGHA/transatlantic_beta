# distutils: language = c++
import numpy as np
cimport numpy as np
from libc.math cimport fabs
import cython
from time import perf_counter
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int8_t, ndim=1] weighted_label_propagation(
    connections, 
    weights, 
    int[::1] initial_labels, 
    size_t max_iterations=100, 
    double tolerance=1e-4):
    """
    Performs weighted label propagation on the data graph represented by `connetions` and `weights`.

    Parameters:
    ----------------
    connections: GraphWrapper
        adjacency list (2D array where each row lists neighbors of a node)
    weights: GraphWrapper
        same shape as connections, stores weights for connections
    initial_labels: iterable
        initial labels for nodes (-1 for unlabeled nodes)
    max_iterations: int
        maximum number of iterations
    tolerance: float
        convergence threshold
    """
    cdef size_t num_nodes = initial_labels.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] scores = np.zeros(num_nodes, dtype=np.float64)
    cdef dict new_scores
    cdef int i, j, neighbor, label, best_label
    cdef double weight, diff, max_diff, score, best_score
    cdef double t_overall = 0.0
    cdef int[::1] final_labels = initial_labels
    cdef np.ndarray[np.int64_t, ndim=1] current_neighbors
    cdef np.ndarray[np.float32_t, ndim=1] current_weights

    for _ in range(max_iterations):
        max_diff = 0.0

        for i in range(num_nodes):
            # Skip already labeled nodes
            if final_labels[i] != -1:  
                continue

            # Initialize new_scores as an empty dictionary
            new_scores = {}

            # Compute scores for all labels based on neighbors
            current_neighbors = connections.to_numpy(i)
            current_weights = weights.to_numpy(i)
            for j in range(current_neighbors.shape[0]):
                neighbor = current_neighbors[j]
                weight = current_weights[j]
                if final_labels[neighbor] != -1:
                    label = final_labels[neighbor]
                    if label in new_scores:
                        new_scores[label] += (1/weight)**2
                    else:
                        new_scores[label] = (1/weight)**2

            # Assign the label with the highest score
            best_label = -1
            best_score = -1.0
            t0 = perf_counter()
            for label, score in new_scores.items():
                if score > best_score:
                    best_score = score
                    best_label = label
            t1 = perf_counter()
            t_overall += t1 - t0

            diff = fabs(scores[i] - best_score)
            max_diff = max(max_diff, diff)
            scores[i] = best_score
            final_labels[i] = best_label

        # Convergence check
        if max_diff < tolerance:
            break

    return np.asarray(final_labels)