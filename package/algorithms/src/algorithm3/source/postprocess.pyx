import numpy as np
cimport numpy as np
cimport cython
from sklearn.metrics import DistanceMetric

# Define the function in Cython
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tdbscan_postprocess(double[:, ::1] points, np.ndarray[int, ndim=1] labels, str metric="minkowski", int p=2):
    cdef int n_points = points.shape[0]
    cdef int n_dims = points.shape[1]
    cdef int i, j, label, lbl
    if metric =="minkowski":
        dist_metric = DistanceMetric.get_metric(metric, p=p)
    else:
        dist_metric = DistanceMetric.get_metric(metric)
    # Calculate centroids for non-negative labels
    unique_labels = np.unique(labels)
    cdef dict labels_to_centroids = {}

    for label in unique_labels:
        if label != -1:
            group = np.array(points)[labels == label]
            centroid = np.mean(group, axis=0)
            labels_to_centroids[label] = centroid

    # Merge labels < -1 with the closest non-negative group
    cdef double min_dist, dist
    cdef int closest_label

    for label in labels_to_centroids:
        if label < -1:
            min_dist = float('inf')
            closest_label = -1
            current_centroid = labels_to_centroids[label]
            for lbl in labels_to_centroids:
                if lbl < -1:
                    continue
                if lbl == label:
                    continue
                compared_centroid = labels_to_centroids[lbl]
                dist = dist_metric.pairwise([compared_centroid],[current_centroid])[0][0]
                if dist < min_dist:
                    min_dist = dist
                    closest_label = lbl
            labels[labels == label] = closest_label

    unique_labels = np.unique(labels)
    labels_to_centroids = {}

    for label in unique_labels:
        if label != -1:
            group = np.array(points)[labels == label]
            centroid = np.mean(group, axis=0)
            labels_to_centroids[label] = centroid

    # Merge points with label -1 (outliers) one by one
    for i in range(n_points):
        if labels[i] == -1:
            min_dist = float('inf')
            closest_label = -1
            for lbl in labels_to_centroids:
                if lbl < -1:
                    continue
                if lbl == label:
                    continue
                compared_centroid = labels_to_centroids[lbl]
                dist = dist_metric.pairwise([compared_centroid],[current_centroid])[0][0]
                if dist < min_dist:
                    min_dist = dist
                    closest_label = lbl
            labels[i] = closest_label

    return labels