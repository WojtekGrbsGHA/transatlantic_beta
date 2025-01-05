import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import scipy
# TODO: dodać możliwosć symetrycznej??

class GraphRepresentation:
    '''
    Retrieval class responsible for generating various representations of graphs.
    Each method of the class relies on all of the class' attributes and takes an additional parameter ```conn_only``` that 
    Attributes
    ----------
    n: int
        Number of vertices of the graph (Also means number of datapoints in the set).
    connections: array-like
        Object that for each index of the first dimension has an ```array-like``` representation of vertices connected to the index.
    weights: 
        Object that for each index of the first dimension has an ```array-like``` representation edges' weights coming out of the 
        index. The order of the edges ***MUST*** be the same as in ```connections``` attribute.

    Methods
    -------
    sparse_matrix_retrieval(conn_only)
        Returns ```scipy.sparse.csr_matrix``` representation of the graph.

    adjacency_matrix_retrieval(conn_only)
        Returns adjacency matrix representation of the graph.

    nx_graph_retrieval(conn_only)
        Returns ```nx.Graph``` representation of the graph.
        
    '''

    def __init__(self, n: int, connections: np.ndarray, weights: np.ndarray):
        self.n = n
        self.connections = connections
        self.weights = weights

    def sparse_matrix(self, conn_only: bool) -> scipy.sparse._csr.csr_matrix:
        '''
        Returns ```scipy.sparse.csr_matrix``` representation of the graph.

        Parameters
        ---------
        conn_only: bool
            Determines whether an edge is represented as ```1``` or weight of the edge.
        '''
        # Number of values to store
        nnz = sum(len(row) for row in self.connections)

        # Declaration of indices for non-empty cells to pass to sparse.csr_matrix
        row_indices = np.empty(nnz, dtype=np.int32)
        col_indices = np.empty(nnz, dtype=np.int32)

        if conn_only:
            data = np.ones(nnz, dtype=np.int8) # 1s for conn_only
        else: 
            data = np.empty(nnz, dtype=np.float32) # weights otherwise, will update them later
        
        # Creating a CSR triplet (V, COL_INDEX, ROW_INDEX)
        # https://en.wikipedia.org/wiki/Sparse_matrix
        idx = 0
        for i in range(self.n):
            row_len = len(self.connections[i])
            row_indices[idx:idx + row_len] = i
            col_indices[idx:idx + row_len] = self.connections[i]
            if not conn_only:
                data[idx:idx + row_len] = self.weights[i] # updating as promised
            idx += row_len
        return csr_matrix((data, (row_indices, col_indices)), shape=(self.n, self.n))
    

    def adjacency_matrix(self, conn_only: bool) -> np.ndarray:
        '''
        Returns an adjacency matrix representation of the graph.

        Parameters
        ---------
        conn_only: bool
            Determines whether an edge is represented as ```1``` or weight of the edge.
        '''
        # Populating with zeros; 1s added later
        adj_matrix = np.zeros((self.n, self.n), dtype=np.float32 if not conn_only else np.int8)

        # Adding 1s
        for i in range(self.n):
            vertices = self.connections[i]
            if not conn_only:
                adj_matrix[i, vertices] = self.weights[i]
            else:
                adj_matrix[i, vertices] = 1

        return np.array(adj_matrix)
    

    def nx_graph(self, conn_only: bool) -> nx.Graph:
        '''
        Returns a ```networkx.Graph``` representation of the graph

        Parameters
        ---------
        conn_only: bool
            Determines whether an edge is represented as ```1``` or weight of the edge.
        '''
        sparse_matrix = self.sparse_matrix(conn_only)
        return nx.from_scipy_sparse_array(sparse_matrix, edge_attribute='weight' if not conn_only else None)