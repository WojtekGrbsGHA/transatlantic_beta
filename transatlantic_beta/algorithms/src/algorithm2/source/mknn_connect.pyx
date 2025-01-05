# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True
import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint all_true(np.ndarray[np.uint8_t, ndim=1] arr):
    cdef Py_ssize_t i, n = arr.shape[0]
    cdef np.uint8_t* data = <np.uint8_t*>arr.data
    
    for i in range(n):
        if data[i] == 0:
            return 0
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple connect_unlabeled(
        np.ndarray knn_connections,
        np.ndarray knn_weights,
        list components,
        np.ndarray[np.uint8_t, ndim=1] vertix_component_label,
        connections, weights):
    """
    Connects unlabeled graph components to labeled components by iteratively 
    finding nearest neighbors and adding connections until all components 
    are connected.

    Parameters:
    --------------
        knn_connections (np.ndarray): 
            2D array of nearest neighbor connections for each node.
        knn_weights (np.ndarray): 
            2D array of edge weights for each nearest neighbor connection.
        components (list): 
            List of components, where each component is a tuple (list of nodes, bool labeled).
        vertix_component_label (np.ndarray[np.uint8_t]): 
            1D array indicating whether each node is part of a labeled component.
        connections: (GraphWrapper)
            A data structure to store graph connections.
        weights: (GraphWrapper)
            A data structure to store the weights of graph edges.
    """
    cdef np.ndarray[np.uint8_t, ndim=1] vertix_component_success = vertix_component_label.copy()

    cdef int idx, node, knn
    cdef double edge_weight
    cdef bint made_labeled, progress
    cdef np.ndarray[np.int64_t, ndim=1] node_knn
    cdef np.ndarray[np.float32_t, ndim=1] node_knn_weights

    cdef list component
    cdef bint labeled

    cdef int c_i, cc_i, n_i, c_length
    cdef int components_length = len(components)

    # Connect unlabeled components directly to labeled nodes
    for c_i in range(components_length):
        # Skip labeled components
        if components[c_i][1]:
            continue
        
        component = components[c_i][0]
        made_labeled = False
        c_length = len(component)
        
        for n_i in range(c_length):
            node = component[n_i]
            node_knn = knn_connections[node]
            node_knn_weights = knn_weights[node]
            
            # Iterate over the node's nearest neighbors
            for idx in range(node_knn.shape[0]):
                knn = node_knn[idx]
                
                # Check if the neighbor is part of a labeled component
                if vertix_component_label[knn]:
                    edge_weight = node_knn_weights[idx]
                    
                    # Update the mutual connections between vertices
                    if not connections.contains(node, knn):
                        connections.append(node, knn)
                        weights.append(node, edge_weight)
                    if not connections.contains(knn, node):
                        connections.append(knn, node)
                        weights.append(knn, edge_weight)
                    
                    # Update success status for the entire component
                    for cc_i in range(c_length):
                        vertix_component_success[component[cc_i]] = True
                        
                    # If the vertix has found its way to the labeled component
                    # Exit both loops because further checks are not necessary 
                    made_labeled = True
                    break
            if made_labeled:
                break

    # Attempt to iteratively connect each vertix to a labeled component
    # May not converge
    while True:
        if all_true(vertix_component_success):
            break
        
        progress = False
        found_way_nodes = set()
        
        # Iterate over each component
        for c_i in range(components_length):
            if components[c_i][1]:
                continue
            
            component = components[c_i][0]
            c_length = len(component)
            
            for n_i in range(c_length):
                node = component[n_i]
                if vertix_component_success[node]:
                    continue
                node_knn = knn_connections[node]
                node_knn_weights = knn_weights[node]
                
                # Iterate over the node's nearest neighbors
                for idx in range(node_knn.shape[0]):
                    knn = node_knn[idx]
                    
                    if vertix_component_success[knn]: # Means that the neighbor is in a labeled component
                        edge_weight = node_knn_weights[idx]
                        
                        # Update the mutual connections between vertices
                        if not connections.contains(node, knn):
                            connections.append(node, knn)
                            weights.append(node, edge_weight)
                        if not connections.contains(knn, node):
                            connections.append(knn, node)
                            weights.append(knn, edge_weight)
                        
                        found_way_nodes.add(node)

                        # If the vertix has found its way to the labeled component
                        # Exit both loops because further checks are not necessary 
                        progress = True
                        break
                if progress:
                    break
            
            # Mark found way nodes as successful 
            # (in a sense of being in a labeled component)
            for nde in found_way_nodes:
                vertix_component_success[nde] = True
        
        # Progress being false means that no vertix has found a labeled-attached neighbor
        # Next iteration would perform exactly the same operations
        if not progress:
            raise SystemError('NO PROGRESS :)')

    return connections, weights