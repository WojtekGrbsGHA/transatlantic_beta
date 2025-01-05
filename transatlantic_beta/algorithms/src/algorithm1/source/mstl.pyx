# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True

from cython cimport boundscheck, wraparound, cdivision
from scipy.spatial import distance
import numpy as np
cimport numpy as np
from scipy.spatial.distance import pdist, squareform
from heapq import heappop, heappush
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t DTYPE32_t
ctypedef np.int64_t ITYPE_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport int64_t
from libc.stddef cimport size_t
from libcpp.unordered_set cimport unordered_set
import numpy as np
cimport numpy as np
cdef enum DataType:
    INT64 = 0
    FLOAT32 = 1

cdef class GraphWrapper:
    cdef:
        void **_data                      # Array of pointers to the data arrays
        int *_sizes                       # Current sizes of the arrays (degrees of vertices)
        bint _lookup                      # If the set is to be made for each connection array for fast lookups
        size_t _count                     # Number of arrays (number of vertices, datapoints)
        DataType _dtype                   # Data type of all arrays (int64 for connections, float32 for weights)
        int _max_size                     # Maximum size of each array (max degree of vertix)
        unordered_set[int64_t] **_int_sets  # For int64 arrays (connections), a hash set per array

    def __cinit__(self, size_t count, DataType dtype, int max_size, bint lookup_opt=False):
        """
        Initialize the structure with a fixed number of arrays.
        All arrays will have the same data type and maximum size.
        Max degree of the vertix is dictated by the max_size, which is k * 3, otherwise algorithm is considered not convergeable.
        """
        cdef size_t i
        self._count = count
        self._dtype = dtype
        self._max_size = max_size * 3
        self._lookup = lookup_opt
        self._data = <void **>malloc(count * sizeof(void *))
        self._sizes = <int *>malloc(count * sizeof(int))

        # Initialize arrays
        for i in range(count):
            self._sizes[i] = 0
            if dtype == DataType.INT64:
                self._data[i] = <void *>malloc(max_size * sizeof(int64_t))
            elif dtype == DataType.FLOAT32:
                self._data[i] = <void *>malloc(max_size * sizeof(DTYPE32_t))
            else:
                raise ValueError("Unsupported data type")

        # Initialize hash sets for int64 arrays
        if dtype == DataType.INT64 and self._lookup:
            self._int_sets = <unordered_set[int64_t] **>malloc(count * sizeof(void *))
            for i in range(count):
                self._int_sets[i] = new unordered_set[int64_t]()
        else:
            self._int_sets = NULL

    cpdef append(self, ITYPE_t index, value):
        """
        Append an element to the array.
        """
        cdef int size
        cdef int updated_size
        cdef void *data
        cdef int64_t *int_data
        cdef DTYPE32_t *double_data
        cdef int64_t int_value
        if not (0 <= index < self._count):
            raise IndexError("Invalid array index")

        if self._sizes[index] >= self._max_size:
            raise OverflowError("Array has reached its maximum size")

        size = self._sizes[index]

        data = self._data[index]

        if size + 1 > self._max_size:
            raise OverflowError("Max vertix degree surpassed!")

        if self._dtype == DataType.INT64:
            int_data = <int64_t *>data
            int_value = <int64_t>value
            int_data[size] = int_value
            if self._lookup:
                self._int_sets[index].insert(int_value)  # Add to hash set for fast lookup
        elif self._dtype == DataType.FLOAT32:
            double_data = <DTYPE32_t *>data
            double_data[size] = <DTYPE32_t>value
        else:
            raise ValueError("Unsupported data type")
        self._sizes[index] += 1

    cpdef tuple pop_max(self):
        """
        Remove and return the maximum value, its list index, and sublist index
        from all sub-arrays. Removes both occurrences of the maximum value.
        """
        cdef list max_indices = []  # List to store all occurrences of the max value
        cdef float max_value_float = -1  # Minimal float64
        cdef int size
        cdef void *data
        cdef float *float_data
        cdef float temp_value_float

        # First pass: Find the maximum value and collect all its indices
        for i in range(self._count):  # Iterate over all lists
            size = self._sizes[i]
            if size == 0:  # Skip empty sublists
                continue

            data = self._data[i]

            if self._dtype == DataType.FLOAT32:
                float_data = <float *> data
                for j in range(size):
                    temp_value_float = float_data[j]
                    if temp_value_float > max_value_float:
                        # Found a new maximum, reset the list of indices
                        max_value_float = temp_value_float
                        max_indices = [(i, j)]
                    elif temp_value_float == max_value_float:
                        # Found another occurrence of the current maximum
                        max_indices.append((i, j))


        if not max_indices:
            raise ValueError("All sub-arrays are empty, cannot pop max")

        # Remove all occurrences of the maximum value
        for i, j in reversed(max_indices):  # Iterate in reverse to avoid shifting issues
            size = self._sizes[i]
            data = self._data[i]
            if self._dtype == DataType.FLOAT32:
                float_data = <float *> data
                # Shift elements left to remove the occurrence
                for k in range(j, size - 1):
                    float_data[k] = float_data[k + 1]
                # if self._lookup:
                #     self._float_sets[i].erase(max_value_float)  # Update lookup
                self._sizes[i] -= 1

        # Return information about the first occurrence
        first_node_index = max_indices[0][0]
        snd_node_index = max_indices[1][0]
        return first_node_index, snd_node_index

    cpdef  pop(self, int list_index, int element_index):
        """
        Remove and return the element at the specified index from a sublist.

        Parameters:
            list_index (int): Index of the list from which to remove the element.
            element_index (int): Index of the element to remove within the sublist.

        Returns:
            int: The removed element.
        """
        cdef int size
        cdef void *data
        cdef int i

        # Validate list_index
        if list_index < 0 or list_index >= self._count:
            raise IndexError("List index out of range")

        # Get the size of the specified sublist
        size = self._sizes[list_index]

        if size == 0:
            raise IndexError("Cannot pop from an empty sublist")

        # Validate element_index
        if element_index < 0 or element_index >= size:
            raise IndexError("Element index out of range")



        data = self._data[list_index]

        if self._dtype == DataType.INT64:
            int_data = <int64_t *>data
            removed_value = int_data[element_index]

            # if self._lookup:
            #     self._int_sets[list_index].remove(removed_value)  # Add to hash set for fast lookup
            for i in range(element_index, size - 1):
                int_data[i] = int_data[i + 1]

        elif self._dtype == DataType.FLOAT32:
            double_data = <DTYPE32_t *>data
            removed_value = double_data[element_index]

            for i in range(element_index, size - 1):
                double_data[i] = double_data[i + 1]

        else:
            raise ValueError("Unsupported data type")

        self._sizes[list_index] -= 1

        return removed_value

    cpdef contains(self, int64_t index, int64_t value):
        """
        Check if a value exists in the array.
        Performable only for `int` arrays for O(1) set performance.
        """
        cdef int64_t target

        if not (0 <= index < self._count):
            raise IndexError("Invalid array index")

        if self._dtype != DataType.INT64:
            raise TypeError("Membership check only supported for int64 arrays")

        target = <int64_t>value
        return self._int_sets[index].count(target) > 0

    cpdef to_numpy(self, ITYPE_t index):
        """
        Convert the specified array to a NumPy array by copying the data from stored arrays.
        """
        cdef int size
        cdef np.ndarray arr
        cdef int64_t *int_data
        cdef DTYPE32_t *double_data

        if not (0 <= index < self._count):
            raise IndexError("Invalid array index")

        size = self._sizes[index]

        if self._dtype == DataType.INT64:
            arr = np.empty(size, dtype=np.int64)
            int_data = <int64_t *>self._data[index]
            memcpy(<void *>arr.data, <void *>int_data, size * sizeof(int64_t))
        elif self._dtype == DataType.FLOAT32:
            arr = np.empty(size, dtype=np.float32)
            double_data = <DTYPE32_t *>self._data[index]
            memcpy(<void *>arr.data, <void *>double_data, size * sizeof(DTYPE32_t))
        else:
            raise ValueError("Unsupported data type")

        return arr

    def to_numpy_all(self):
        """
        Convert all arrays to a list of NumPy arrays.
        """
        cdef int i
        cdef list result = []

        for i in range(self._count):
            result.append(self.to_numpy(i))

        return result

    def __dealloc__(self):
        cdef size_t i

        for i in range(self._count):
            if self._data[i] != NULL:
                free(self._data[i])

        free(self._data)
        free(self._sizes)

        # Deallocate hash sets for int64 arrays
        if self._dtype == DataType.INT64 and self._int_sets != NULL:
            for i in range(self._count):
                if self._int_sets[i] != NULL:
                    del self._int_sets[i]
            free(self._int_sets)




@boundscheck(False)
@wraparound(False)
@cdivision(False)
def prim_mst(np.ndarray[np.float64_t, ndim=2] X, str metric='euclidean'):
    '''
    Performs Prim's algorithm on an array of datapoints, creating a minimum spanning tree
    using a complete weighted graph.

    Parameters
    ----------
    X: np.ndarray
        Array of datapoints in a continuous or discrete space.
    metric: str
        Distance metric as defined in scipy.spatial.distance.pdist.

    Returns
    ----------
    tuple
        A 2-tuple containing:
        - mst_connections: list of lists, representing the adjacency list of the MST.
        - mst_weights: list of lists, representing the edge weights of the MST.
    '''
    cdef int n = X.shape[0]

    # Pairwise distances
    cdef np.ndarray[DTYPE_t, ndim=1] distances = pdist(X, metric=metric)
    cdef np.ndarray[DTYPE_t, ndim=2] distance_matrix = squareform(distances)

    mst_connections = GraphWrapper(n, DataType.INT64, n)
    mst_weights = GraphWrapper(n, DataType.FLOAT32, n)

    cdef np.ndarray[np.uint8_t, ndim=1] visited = np.zeros(n, dtype=np.bool_)
    cdef list min_heap = [(0.0, 0, -1)]  # (weight, current_node, parent_node)

    cdef double weight
    cdef int u, v, parent

    # Prim's algorithm
    while min_heap:
        weight, u, parent = heappop(min_heap)

        if visited[u]:
            continue

        visited[u] = True

        if parent != -1:  # All cases outside of the first node
            mst_connections.append(u, parent)
            mst_connections.append(parent, u)
            mst_weights.append(u, weight)
            mst_weights.append(parent, weight)

        for v in range(n):
            if not visited[v]:
                heappush(min_heap, (distance_matrix[u, v], v, u))

    return mst_connections, mst_weights

@boundscheck(False)
@wraparound(False)
@cdivision(False)
def find_all_paths_with_labels(
    GraphWrapper connections_sparse,
    np.ndarray[np.int8_t, ndim=1] labels,
    int start,
    int bomb_node=-1,
    path=None
):
    """
    Find all paths from the starting node to nodes with labels greater than -1 using DFS algorithm.
    Stop searching a path if a node with the same label as the starting node is encountered,
    if a node with a label > -1 is found, or if a "bomb node" is encountered.

    Parameters
    ----------
    connections_sparse: np.ndarray
        Object that for each index of the first dimension has a list representation of vertices connected to the index.
    labels: np.ndarray
        Array in which for each index that refers to node there is label information.
    start: int
        Index of starting node.
    bomb_node: int, optional
        Index of the "bomb node". If visited, the search in this direction stops. Defaults to None.
    path:
        Path at the current state of algorithm. Defaults to None.

    Returns
    ----------
    List of all paths from the starting node to nodes with labels > -1
    """

    cdef list result_paths = []
    cdef list current_path

    if path is None:
        current_path = [start]
    else:
        current_path = path + [start]

    if bomb_node != -1 and start == bomb_node:
        return []

    if labels[start] > -1 and start != current_path[0]:
        return [current_path]

    cdef np.ndarray[ITYPE_t, ndim=1] neighbors = connections_sparse.to_numpy(start)
    cdef int neighbor
    for i in range(len(neighbors)):
        neighbor = neighbors[i]
        if neighbor not in current_path:
            # Skip neighbors with the same label as the starting node
            if labels[neighbor] == labels[current_path[0]]:
                continue

            # Recursively find paths
            result_paths.extend(
                find_all_paths_with_labels(
                    connections_sparse=connections_sparse,
                    labels=labels,
                    start=neighbor,
                    bomb_node=bomb_node,
                    path=current_path
                )
            )

    return result_paths





@boundscheck(False)
@wraparound(False)
@cdivision(False)
def explore_graph_recursive(GraphWrapper connections_sparse, np.ndarray[np.int8_t, ndim=1] labels, int start):
    """
    Explore the graph recursively by finding all paths from a starting node.
    After finding paths, recursively call the function for the end nodes of each path,
    passing the current start as a bomb node to prevent backtracking.

    Parameters
    ----------
    connections_sparse: np.ndarray
        Graph adjacency list in sparse format.
    labels: np.ndarray
        Array of labels for each node.
    start: int
        Starting node.

    Returns
    ----------
    List of all paths explored during the traversal.
    """


    def recursive_explore(int current_start, int bomb_node, list visited_paths,  np.ndarray[np.int8_t, ndim=1] labels):

        cdef list paths = find_all_paths_with_labels(connections_sparse, labels, current_start, bomb_node=bomb_node)

        visited_paths.extend(paths)


        cdef list path
        cdef int end_node
        for i in range(len(paths)):
            path = paths[i]
            end_node = path[len(path) - 1]
            if end_node != bomb_node:
                new_bomb_node = path[len(path) - 2]
                recursive_explore(current_start=end_node, bomb_node=new_bomb_node, visited_paths=visited_paths, labels=labels)


    cdef list all_paths = []

    recursive_explore(current_start=start, bomb_node=-1, visited_paths=all_paths, labels=labels)

    return all_paths

@boundscheck(False)
@wraparound(False)
@cdivision(False)
def find_subgraph_with_labels(GraphWrapper connections_sparse,
                              GraphWrapper weights_sparse,
                              np.ndarray[np.int8_t] labels):
    """
    Creates subgraph of original graph that consists of paths between nodes with different labels.
    If node is not labeled, it is not treated as an obstacle for path.

    Parameters
    ----------
    weights_sparse
    connections_sparse
    labels: np.ndarray
        Array in which each index refers to a node and its label information.

    Returns
    -------
    sub_connections_sparse, sub_weights_sparse: np.ndarray
        Objects similar to connections_sparse and weights_sparse that define a subgraph of the original graph.
    """
    cdef list all_paths = []
    cdef list node_list, unvisited_labeled_nodes
    cdef set nodes_visited = set()
    cdef int start_node
    cdef list new_paths

    node_list = [i for i in range(len(labels)) if labels[i] != -1]
    unvisited_labeled_nodes = node_list.copy()

    while unvisited_labeled_nodes:
        start_node = unvisited_labeled_nodes.pop(0)
        new_paths = explore_graph_recursive(connections_sparse, labels, start_node)

        if len(new_paths) > 0:
            for i in range(len(new_paths)):
                path = new_paths[i]
                all_paths.append(path)
            for i in range(len(all_paths)):
                path = all_paths[i]
                nodes_visited.update(path)
        else:
            nodes_visited.add(start_node)

        unvisited_labeled_nodes = []
        for i in range(len(node_list)):
            node = node_list[i]
            if node not  in nodes_visited:
                unvisited_labeled_nodes.append(node)

    cdef int n = len(connections_sparse.to_numpy_all())
    sub_connections_sparse = GraphWrapper(n, DataType.INT64, n)
    sub_weights_sparse = GraphWrapper(n, DataType.FLOAT32, n)

    # Populate sub_connections_sparse and sub_weights_sparse based on the paths
    cdef int node_a, node_b
    cdef int edge_index
    cdef float edge_value

    for i in range(len(all_paths)):
        path = all_paths[i]
        for j in range(len(path) - 1):
            node_a = path[j]
            node_b = path[j + 1]

            if node_b not in sub_connections_sparse.to_numpy(node_a):
                sub_connections_sparse.append(node_a, node_b)
                edge_index = list(connections_sparse.to_numpy(node_a)).index(node_b)
                edge_value = weights_sparse.to_numpy(node_a)[edge_index]
                sub_weights_sparse.append(node_a, edge_value)

            if node_a not in sub_connections_sparse.to_numpy(node_b):
                sub_connections_sparse.append(node_b, node_a)
                edge_index = list(connections_sparse.to_numpy(node_b)).index(node_a)
                edge_value = weights_sparse.to_numpy(node_b)[edge_index]
                sub_weights_sparse.append(node_b, edge_value)

    return sub_connections_sparse, sub_weights_sparse

@boundscheck(False)
@wraparound(False)
@cdivision(False)
def make_labels(
    GraphWrapper connections_sparse,
    int node,
    int label,
    set visited,
    np.ndarray[np.int8_t, ndim=1] labels
):
    """
    Recursively assigns the same label to all nodes connected to the starting node.
    Modifies the labels array in place.

    Parameters
    ----------
    node: int
        Index of starting node.
    connections_sparse: list
        Adjacency list where each index has a list of neighbors.
    label: int
        Label of the starting node.
    visited: set
        Set of visited nodes to avoid redundant processing.
    labels: np.ndarray[np.int32_t, ndim=1]
        Array where each index represents a node and stores its label.
    """
    cdef int neighbor

    # Mark the current node as visited
    visited.add(node)
    labels[node] = label

    neighbor_list = connections_sparse.to_numpy(node)
    for i in range(len(neighbor_list)):
        neighbor = neighbor_list[i]
        if neighbor not in visited:
            make_labels(connections_sparse, neighbor, label, visited, labels)


@boundscheck(False)
@wraparound(False)
@cdivision(False)
def clusterize(GraphWrapper connections_sparse,
               GraphWrapper weights_sparse,
               np.ndarray[np.int8_t, ndim=1] labels):
    """
    Main function for clustering the data with given labels. The algorithm is a modification of
    traditional edge cutting in the MST graph. It finds a subgraph of edges that are part of the path
    that connects nodes of different labels. It chooses edges from that subgraph and cuts them until
    all nodes with different labels are separated. At the end, it connects the nodes with the same labels.

    Parameters
    ----------
    weights_sparse: np.ndarray
        Array of edge weights.
    connections_sparse: np.ndarray
        Array of connections (edges) between nodes.
    labels: np.ndarray
        Array where each index represents a node and its corresponding label.

    Returns
    -------
    new_labels: np.ndarray
        Array of new labels, where all nodes are assigned to a label.
    """

    cdef GraphWrapper sub_connections_sparse
    cdef GraphWrapper sub_weights_sparse
    cdef int break_flg
    cdef int n = len(labels)
    cdef dict labels_dict = {}
    cdef list same_labels_groups = []
    cdef list cluster_representative_nodes = []
    cdef list group
    cdef int i, label

    while True:
        # Find the subgraph with the labels
        sub_connections_sparse, sub_weights_sparse = find_subgraph_with_labels(connections_sparse, weights_sparse, labels)

        # If no edges were found, stop the process
        paths = sub_connections_sparse.to_numpy_all()
        break_flg = 1

        for i in range(len(paths)):
            sublist = paths[i]
            if len(sublist) > 0:
                break_flg = 0


        if break_flg == 1:
            break

        first_node_index, snd_node_index = sub_weights_sparse.pop_max()
        for_first_idx = list(connections_sparse.to_numpy(first_node_index)).index(snd_node_index)
        for_snd_idx = list(connections_sparse.to_numpy(snd_node_index)).index(first_node_index)

        connections_sparse.pop(first_node_index, for_first_idx)
        connections_sparse.pop(snd_node_index, for_snd_idx)
        weights_sparse.pop(first_node_index, for_first_idx)
        weights_sparse.pop(snd_node_index, for_snd_idx)


    for i in range(n):
        label = labels[i]
        if label != -1:
            if label not in labels_dict:
                labels_dict[label] = []
            labels_dict[label].append(i)

    cdef list labels_groups = [group for group in labels_dict.values()]
    for i in range(len(labels_groups)):
        group = labels_groups[i]
        if len(group) > 1:
            same_labels_groups.append(group)
            for i in range(len(group) - 1):
                start_node = group[i]
                end_node = group[i + 1]
                connections_sparse.append(start_node, end_node)
                connections_sparse.append(end_node, start_node)
                weights_sparse.append(start_node, 1)
                weights_sparse.append(end_node, 1)

    for i in range(len(labels_groups)):
        group = labels_groups[i]
        cluster_representative_nodes.append(group[0])
    new_labels = labels.copy()

    for i in range(len(cluster_representative_nodes)):
        node = cluster_representative_nodes[i]
        make_labels(connections_sparse, node, new_labels[node], visited=set(), labels=new_labels)

    return connections_sparse, weights_sparse, new_labels





