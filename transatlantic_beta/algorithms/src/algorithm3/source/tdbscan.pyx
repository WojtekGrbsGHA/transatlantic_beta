import numpy as np
cimport numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
cimport cython

cdef extern from "limits.h":
    cdef int INT_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
def semi_supervised_dbscan(double[:, ::1] data,
                           double eps,
                           int min_samples,
                           int[::1] labels,
                           str metric,
                           int p):
    """
    Semi-supervised DBSCAN implementation.

    Parameters:
    ----------
    data : double[:, ::1]
        2D NumPy array of shape (n_samples, n_features)
    eps : doubel
        Epsilon neighborhood radius
    min_samples : int
        Minimum number of samples to form a core point
    labels : int[::1]
        1D NumPy array of shape (n_samples,) with labeled data.
        labels[i] >= 0 indicates the point is labeled.
        labels[i] == -1 indicates the point is unlabeled.
    metric : str
        Metric defining the distance between data points.
    p : int
        Defines which Minkowski p-norm to use. Works only if metric is set to `minkowski`.

    Returns:
    ----------
    cluster_labels : np.ndarray
        1D NumPy array of cluster assignments for each point.
    """
    cdef int n_samples = data.shape[0]

    # Cluster labels set to `-1` by default
    cdef np.ndarray[int, ndim=1] cluster_labels = np.full(n_samples, -1, dtype=np.int32)
    cdef np.ndarray[char, ndim=1] visited = np.zeros(n_samples, dtype=np.int8)

    cdef list[list] tree_ball_proximity

    # Building KD-tree [SciPy] or BallTree [sklearn] depending on the metric
    if metric == "minkowski" and p == 2:
        # For l2 norm using SciPy's implementation for performance
        tree = KDTree(data)
        tree_ball_proximity = tree.query_ball_tree(tree, r=eps)
    else:
        if metric == "minkowski":
            tree = BallTree(data, metric=metric, p=p) # p defines minkowski's p-norm, additional kwarg
        else:
            tree = BallTree(data, metric=metric)
        tree_ball_proximity = tree.query_radius(data, r=eps)

    # Negative means valid but not true
    cdef int cluster_id = -2

    # Convert arrays to memoryviews for efficiency
    cdef double[:, :] data_view = data
    cdef int[::1] labels_view = labels
    cdef int[::1] cluster_labels_view = cluster_labels
    cdef char[::1] visited_view = visited
    cdef char[::1] soft_visited_view = visited

    cdef int i

    for i in range(n_samples):
        if soft_visited_view[i]:
            continue

        if labels_view[i] >= 0 and (len(tree_ball_proximity[i]) < min_samples):
            # Labeled point, treat as core point
            expand_cluster(data_view, min_samples, labels_view,
                           cluster_labels_view, soft_visited_view, i, labels_view[i], tree_ball_proximity)
            
            # cluster_id += 1
        
        # No visitation update for non-core 
    for i in range(n_samples):
        if visited_view[i]:
            continue

        if labels_view[i] >= 0 and (len(tree_ball_proximity[i]) >= min_samples):
            # Labeled point, treat as core point
            expand_cluster(data_view, min_samples, labels_view,
                           cluster_labels_view, visited_view, i, labels_view[i], tree_ball_proximity)
    for i in range(n_samples):
        if visited_view[i]:
            continue
        # Unlabeled point
        neighbors = tree_ball_proximity[i]
        if len(neighbors) >= min_samples:
            expand_cluster(data_view, min_samples, labels_view,
                            cluster_labels_view, visited_view, i, cluster_id, tree_ball_proximity)
            cluster_id -= 1
        else:
            # Noise or border point, mark as visited
            visited_view[i] = 1

    return np.asarray(cluster_labels)
def expand_cluster(double[:, :] data,
                    int min_samples,
                    int[::1] labels,
                    int[::1] cluster_labels,
                    char[::1] visited,
                    Py_ssize_t point_idx,
                    int cluster_id,
                    list tree_ball_proximity):

    cdef int n_samples = data.shape[0]
    cdef int i, idx, n_neighbors
    cdef int queue_start = 0, queue_end = 0

    # Initialize a queue with maximum size
    cdef int[::1] queue = np.empty(INT_MAX, dtype=np.int32)
    queue[queue_end] = point_idx
    queue_end += 1

    while queue_start < queue_end:
        current_point_idx = queue[queue_start]
        queue_start += 1

        if visited[current_point_idx] and (cluster_labels[current_point_idx] != -1):
            continue

        # Mark point as visited and assign cluster
        visited[current_point_idx] = 1
        cluster_labels[current_point_idx] = cluster_id

        # Get neighbors
        neighbors = tree_ball_proximity[current_point_idx]
        n_neighbors = len(neighbors)

        # Core point, it propagates further to all neighbors
        if (labels[current_point_idx] >= 0) or (n_neighbors >= min_samples):
            # Core point
            for i in range(n_neighbors):
                idx = neighbors[i]
                # if (not visited[idx] or (visited[idx] and (cluster_labels[idx] != -1))) and ((labels[idx] < 0) or (labels[idx] == cluster_id)):
                if ((labels[idx] < 0) or (labels[idx] == cluster_id)):
                    queue[queue_end] = idx
                    queue_end += 1
        else:
            # Border point (already assigned cluster label)
            pass
    del queue