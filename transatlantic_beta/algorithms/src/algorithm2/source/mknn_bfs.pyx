# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int64_t ITYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple bfs(int start_node, np.uint8_t[::1] visited, np.uint8_t[::1] is_labeled, connections):
    """
    Perform a Breadth-First Search (BFS) on a graph starting from a given node and noting whether it contains
    at least one labeled datapoint.

    Parameters
    ----------
    start_node : int
        The starting node for the BFS traversal.
    visited : np.ndarray
        1D array of type `np.uint8_t` where each element indicates whether a node has been visited (1 for visited, 0 otherwise).
    is_labeled : np.ndarray
        1D array of type `np.uint8_t` where each element indicates whether a node is labeled (1 for labeled, 0 otherwise).
    connections : GraphWrapper
        An object representing the graph structure.

    Returns
    ----------
    tuple
        - component : list[int]
            A list of nodes that form the connected component together with `start_node`.
        - is_labeled_component : int
            A flag (1 or 0) indicating whether any node in the connected component is labeled.

    Notes
    ----------
    - The function assumes that `visited` and `is_labeled` arrays have a length equal to the total number of nodes in the graph.
    """
    cdef int num_nodes = visited.shape[0]
    cdef np.ndarray[ITYPE_t, ndim=1] queue = np.empty(num_nodes, dtype=np.int64)
    cdef int front = 0
    cdef int back = 0
    queue[back] = start_node
    back += 1
    visited[start_node] = 1
    cdef list component = []
    cdef int is_labeled_component = 0 if not is_labeled[start_node] else 1
    cdef int node
    cdef int neighbor, i, neighbor_n
    cdef np.ndarray[ITYPE_t, ndim=1] neighbors

    while front < back:
        node = queue[front]
        front += 1
        component.append(node)
        neighbors = connections.to_numpy(node)
        neighbor_n = neighbors.shape[0]
        for i in range(neighbor_n):
                neighbor = neighbors[i]
                if visited[neighbor] == 0:
                    visited[neighbor] = 1
                    if is_labeled_component == 0 and is_labeled[neighbor]:
                        is_labeled_component = 1
                    queue[back] = neighbor
                    back += 1
    return component, is_labeled_component
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple find_connected_components(X, connections):
    """
    Finds the connected components in a graph and flags them based on whether there are any 
    labeled datapoint associated to them.

    Parameters
    ----------
    X : _AlgorithmInput
        Input data object containing node information and methods to access labeled indices 
        and other metadata.
    
    connections : GraphWrapper
        An object representing the graph structure.

    Returns
    -------
    tuple
        A tuple containing:
        - components (list of tuples): A list of connected components where each component is 
          represented as a tuple `(nodes, is_labeled)`. `nodes` is a list of node indices in the 
          component, and `is_labeled` is a flag if the component contains at least one labeled datapoint.
        - vertix_component_label (np.ndarray): A 1D numpy array where the value at each index 
          corresponds to the flag whether the vertix belongs to a component with at least labeled datapoint.
    """
    cdef int num_nodes = X.n
    cdef np.uint8_t[::1] visited = np.zeros(num_nodes, dtype=np.uint8)
    cdef np.uint8_t[::1] is_labeled = np.zeros(num_nodes, dtype=np.uint8)
    cdef np.int64_t[::1] labeled_indices = X.labeled_indices
    cdef int i
    cdef int labeled_count = X.labeled_count
    for i in range(labeled_count):
        is_labeled[labeled_indices[i]] = 1
    cdef int idx
    cdef list components = []
    cdef np.ndarray[np.uint8_t, ndim=1] vertix_component_label = np.zeros(num_nodes, dtype=np.uint8)

    cdef int start_node = 0
    cdef tuple new_component

    # Iterate over all nodes to find connected components
    for start_node in range(num_nodes):
        if visited[start_node] == 0:
            new_component = bfs(start_node, visited, is_labeled, connections)
            for idx in new_component[0]:
                vertix_component_label[idx] = new_component[1]
            components.append(new_component)
    return components, vertix_component_label
