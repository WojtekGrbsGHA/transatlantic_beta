# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True
from cython cimport boundscheck, wraparound, cdivision
from scipy.spatial import distance
import numpy as np
cimport numpy as np
from scipy.spatial import KDTree
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t DTYPE32_t
ctypedef np.int64_t ITYPE_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport int64_t
from libc.stddef cimport size_t
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
import time
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
                self._data[i] = <void *>malloc(self._max_size * sizeof(int64_t))
            elif dtype == DataType.FLOAT32:
                self._data[i] = <void *>malloc(self._max_size * sizeof(DTYPE32_t))
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
@cdivision(True)
cpdef tuple mutual_knn(double[:, ::1] X, int k, int p=2):
    """
    Constructs a mutual k-nearest neighbors graph.
    
    Parameters
    ----------
    X : double[:, ::1]
        Data for which the mutual k-nearest neighbors graph will be constructed.
    k : int
        Number of nearest neighbors.
    p : int
        Which Minkowski p-norm to use. A large, finite p may cause a ValueError if overflow occurs.
    
    Returns
    -------
    connections_sparse : np.ndarray
        Indices of mutual nearest neighbors.
    weights_sparse : np.ndarray
        Distances to mutual nearest neighbors.
    """
    cdef int n = X.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] knn_weights_sparse_handler
    cdef np.ndarray[DTYPE32_t, ndim=2] knn_weights_sparse
    cdef np.ndarray[ITYPE_t, ndim=2] knn_connections_sparse

    # KDTree for efficient nearest neighbor search
    _kd_tree = KDTree(X)
    knn_weights_sparse_handler, knn_connections_sparse = _kd_tree.query(X, k=k + 1, workers=-1)
    
    # Remove self-connections
    knn_weights_sparse = knn_weights_sparse_handler[:, 1:].astype(np.float32)
    knn_connections_sparse = knn_connections_sparse[:, 1:]
    
    # Casting the arrays to cdefined class
    mutual_connections = GraphWrapper(n, DataType.INT64, k) # No fast lookup needed for knn
    mutual_weights = GraphWrapper(n, DataType.FLOAT32, k)

    cdef ITYPE_t[::1] curr_neighbor
    cdef int i, j, l

    # Searching for mutual knn connections
    for i in range(n):
        for j in range(k):
            neighbor = knn_connections_sparse[i, j]
            # Check if mutual (i in neighbor's k-nearest neighbors)
            curr_neighbor = knn_connections_sparse[neighbor]
            for l in range(curr_neighbor.shape[0]):

                # If they both contain each other in the knn list
                if i == curr_neighbor[l]:
                    mutual_connections.append(i, neighbor)
                    mutual_weights.append(i, knn_weights_sparse[i, j])
    return mutual_connections, mutual_weights, knn_connections_sparse, knn_weights_sparse

@boundscheck(False) 
@wraparound(False)   
@cdivision(True)
cpdef unordered_set[int] _find_idx_set(const DTYPE32_t[:, :] distances, ITYPE_t[::1] labels, np.ndarray[ITYPE_t, ndim=1] connections, np.float32_t[::1] weights):
    """
    _find_idx_set: Function to find a set of indices based on distance, labels, connections, and weights. 

    Parameters:
    -----------
    distances : DTYPE32_t[:, :]
        Pairwise distances between elements. 
        Assumes distances[j, j] == 0 for all `j`.

    labels : ITYPE_t[::1]
        IMemview of indices of data points that are labeled.

    connections : np.ndarray[ITYPE_t, ndim=1]
        Each entry corresponds to a row index in the distances matrix.

    weights : np.float32_t[::1]
        weights corresponding to each connection.
    """
    cdef int rows = connections.shape[0]
    cdef int cols = labels.shape[0]
    cdef int i,j,min_column_index
    cdef int prev_min_index = -1
    cdef int d_i, d_j
    cdef unordered_set[int] i_set
    cdef double current_value
    cdef np.float32_t min_column_value

    for j in range(cols):
        min_column_value = INFINITY
        d_j = labels[j]
        for i in range(rows):
            d_i = connections[i]
            # distances[j,j] == 0 so the j is argmin no matter the values of w[i]
            # if d_i == d_j:
            #     min_column_index = i
            #    break
            # It slows it down, maybe useful for high count of labeled points 
            current_value = distances[d_i, d_j] + weights[i]
            if current_value < min_column_value:
                min_column_index = i
                min_column_value = current_value
        # Slight optimization, reduces the insert overhead (it is implemented due to empirical evidence of this case often occuring)
        if prev_min_index == min_column_index:
            continue
        else:
            prev_min_index = min_column_index
        
        i_set.insert(min_column_index)

        if i_set.size() == cols:
            break
    return i_set      

def create_pairwise_distance_matrix(X, filename, chunk_size=5000):
    """
    Create a pairwise distance matrix and store it on disk using a memory-mapped file.
    
    Parameters:
    -----------
    X: _AlgorithmInput
        Input data object containing node information and methods to access labeled indices 
        and other metadata.
    filename: str
        Path to the memory-mapped file.
    chunk_size: int 
        Number of rows to process in chunks.
    """
    cdef int n = X.n
    cdef int n_labeled = X.labeled_count
    cdef np.ndarray[DTYPE32_t, ndim=2] data = X.data
    cdef np.ndarray[DTYPE32_t, ndim=2] data_labeled = X.get_labeled()
    # Create a memory-mapped file for the distance matrix
    cdef np.ndarray[DTYPE32_t, ndim=2] distance_matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(n, n_labeled))
    
    # Process in chunks to calculate distances row by row
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
            
        # Extract sub-matrices for processing
        chunk_i = data[i:end_i]
        
        # Calculate pairwise distances for the current chunks
        dists = distance.cdist(chunk_i, data_labeled, 'euclidean')
        
        # Store the computed distances in the memory-mapped file
        distance_matrix[i:end_i, :] = dists

    # Ensure the file is saved
    distance_matrix.flush()

@boundscheck(False) 
@wraparound(False)
@cdivision(True)  
cdef DTYPE32_t[:, :] _pairwise_fast(np.ndarray[ITYPE_t, ndim=1] label_indices, X):
    """
    _pairwise_fast: Function returning a pairwise distances matrix of shape `(n, n_l)` where n is the number of datapoints
                    and n_l is the number of labeled datapoints. Returns a float32 memview of the distances cached in RAM.

    Parameters:
    -----------
    label_indices: np.ndarray
        NumPy array containing indices of labeled points in the dataset
    X: _AlgorithmInput
        Input data object containing node information and methods to access labeled indices 
        and other metadata.
    """
    cdef DTYPE32_t[:, :] pairwise_array = distance.cdist(X.data, X.get_labeled(), 'euclidean').astype(np.float32)
    return pairwise_array
@boundscheck(False) 
@wraparound(False)   
@cdivision(True)
cpdef informative_edges(X, int k, connections: GraphWrapper , weights: GraphWrapper):
    """
    Parameters:
    -----------
    X : _AlgorithmInput
        Input data object containing node information and methods to access labeled indices 
        and other metadata.

    k : int
        The maximum number of connections to retain for each node.

    connections : GraphWrapper
        An object representing the pre-existing **mknn** connections between nodes.

    weights : GraphWrapper
        An object representing the weights associated with the connections in the 
        `connections` array.
    
    Returns:
    -----------
    connections_after : GraphWrapper
        Container of pointer arrays of connections associated with each vertix, such that the edges
        minimize the **sum of distances between each other and at least one labeled** point in the dataset.

    weights after : GraphWrapper
        Container of pointer arrays of weights associated with the `connections`.
    """

    cdef int n = X.n
    cdef np.ndarray[ITYPE_t, ndim=1] labeled_arange = np.arange(X.labeled_count, dtype=np.int64)
    cdef const DTYPE32_t[:, :] distance_matrix

    # Greater than 20_000 is considered not-cachable, performing a memory map on the drive.
    if n > 30_000:
        create_pairwise_distance_matrix(X, 'tmp.dat')
        distance_matrix = np.memmap('tmp.dat', dtype='float32', mode='r', shape=(n,n))
    else:
        distance_matrix = _pairwise_fast(labeled_arange, X)
    # Random shuffle for potential performance gains. Points from the same label are less likely
    # to make the algorithm keep all of the edges (and therefore breaking out of the set creating loop).
    np.random.shuffle(labeled_arange)

    cdef int i, j, connections_no, label

    cdef np.float32_t[::1] weight_array
    cdef np.ndarray[ITYPE_t, ndim=1] connection_array

    # Set of indices that satisfy the argmin condition
    cdef unordered_set[int] connections_indices_set
    cdef unordered_set[int].iterator it
    
    # Casting weights and edges onto the arrays of pointers
    weights_after = GraphWrapper(n, DataType.FLOAT32, k)
    # Fast lookup is required for later checks if a connection is already registered
    connections_after = GraphWrapper(n, DataType.INT64, k, lookup_opt=True) 

    cdef np.int8_t idx
    cdef double total_time
    t_total = 0
    # Iterating over every node to find which mknn neighbors satisfy the argmin condition
    for i in range(n):
        if connections._sizes[i] == 0:
            continue
        connection_array = connections.to_numpy(i)
        weight_array = weights.to_numpy(i)
        connections_no = connection_array.shape[0]
        # Set of indices that satisfy the argmin condition for vertix `i`
        connections_indices_set = _find_idx_set(distance_matrix, labeled_arange, connection_array, weight_array) 
        # Iterating over the set and updating the connections and weights
        it = connections_indices_set.begin()
        while it != connections_indices_set.end():
            idx = deref(it)
            connected_vertix = connection_array[idx]
            weight_vertix = weight_array[idx]

            # The vertix might already have the neighbor registered
            if not connections_after.contains(i, connected_vertix): 
                connections_after.append(i, connected_vertix)
                weights_after.append(i, weight_vertix)

            # The neighbor might already have the vertix registered
            if not connections_after.contains(connected_vertix, i):
                connections_after.append(connected_vertix, i)
                weights_after.append(connected_vertix, weight_vertix)
            inc(it)
    return connections_after, weights_after