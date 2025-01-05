from ..preprocess.preprocessing import _preprocess_ndarray_matrix
from ..graph.graph_retrieval import GraphRetrieval

from collections import defaultdict
import uuid
import numpy as np
import sys
import scipy
from scipy.spatial import KDTree
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from heapq import heappop, heappush
import matplotlib.pyplot as plt

# TODO: Uzupełnić docstringi dla klas

class DataGraph:
    '''
    Base class for representing data points as a graph. Further extensions of the class implement various ways of the graphs' construction,
    such as ```KNN``` or ```MST```. 

    Attributes
    ----------
    connections_sparse: np.ndarray
        Object that for each index of the first dimension has a list representation of vertices connected to the index.
    weights_sparse: np.ndarray
        Object that for each index of the first dimension has a list representation edges' weights coming out of the 
        index. The order of the edges ***MUST*** be the same as in ```connections_sparse``` attribute.

    Internal Attributes
    ----------
    _n_samples: int
        Size of the dataset the graph was fitted to.
    _fit: bool
        Flag that tells if the graph has already been fitted on any data.

    Methods
    ----------
    # TO BE ADDED
    '''
    graph_fittnes_validation_string = 'Fit the graph to the data first or use the generation function: generate_'

    def __init__(self):
        self._fit = False

    def fit(self, X: np.ndarray, **kwargs: any):
        '''
        Fits a desired graph to the data. If invoked on an instance of ```DataGraph```, nothing will happen.


        Parameters
        ----------
        X: array-like
            Array of datapoints the graph will be fitted on. ***Data points represent the rows***, ***Columns are the dimensions***.
        **kwargs: any
            Further arguments, passed to ```_generate_graph_source``` of a particular child class.
        '''
        X = self._assert_input_data(X)
        self._generate_graph_source(X, **kwargs)
        self._fit = True
        self.graph_retrieval = GraphRetrieval(n=self._n_samples,
                                              connections=self.connections_sparse,
                                              weights=self.weights_sparse)

    # Source Graph
    # Source means 2-tuple of (distances, indices) of retrieved graph based on the input data.
    # TERAZ SOURCE JEST TAKI DZIWNY BO TAKI JEST NAJLEPSZY DLA KNN I NAJMNIEJ WAZY, NIE MA OGROMNYCH RZECZY
    # JEZELI NA COS ZAMIENIAC TO NA SPARSE MATRIX (ALBO NX GRAPH ALE ON TEZ POTRZEBUJE WTEDY SPARSE DO UTWORZENIA SIEBIE)
    def _generate_graph_source(self):
        pass

    ######################## Sparse matrices ########################
    def generate_sparse_matrix(self, X: np.ndarray, conn_only: bool = True):
        '''
        Generate a ```scipy.sparse.crs_matrix``` representation of the graph based on the data X.

        Parameters
        ----------
        X: array-like
            Input data as a (M x N) matrix of points, row being a datapoint and columns being the dimensions.
        conn_only: bool
            Determines whether an edge is represented as ```1``` or weight of the edge.
        '''
        if not self.is_fitted():
            self.fit(X)
        return self.sparse_matrix(conn_only)

    def sparse_matrix(self, conn_only: bool = False) -> scipy.sparse._csr.csr_matrix:
        self._validate_retrieval_fitness('sparse_matrix()')
        return self.graph_retrieval.sparse_matrix_retrieval(conn_only=conn_only)

    ################################################################

    ######################## nx Graphs ########################
    def generate_nx_graph(self, X: np.ndarray, conn_only: bool = True) -> nx.Graph:
        '''
        Generate a ```networkx.Graph``` representation of the graph based on the data X.

        Parameters
        ----------
        X: array-like
            Input data as a (M x N) matrix of points, row being a datapoint and columns being the dimensions.
        conn_only: bool
            Determines whether an edge is represented as ```1``` or weight of the edge.
        '''
        if not self.is_fitted():
            self.fit(X)
        return self.nx_graph(conn_only)

    def nx_graph(self, conn_only: bool = False) -> nx.Graph:
        self._validate_retrieval_fitness('nx_graph()')
        return self.graph_retrieval.nx_graph_retrieval(conn_only=conn_only)

    ################################################################

    ######################## Adjacency matrix ########################
    def generate_adjacency_matrix(self, X: np.ndarray, conn_only: bool = True) -> np.ndarray:
        '''
        Generate an adjacency matrix representation of the graph based on the data X.

        Parameters
        ----------
        X: array-like
            Input data as a (M x N) matrix of points, row being a datapoint and columns being the dimensions.
        conn_only: bool
            Determines whether an edge is represented as ```1``` or weight of the edge.
        '''
        if not self.is_fitted():
            self.fit(X)
        return self.adjacency_matrix(conn_only)

    def adjacency_matrix(self, conn_only: bool = True):
        self._validate_retrieval_fitness('adjacency_matrix()')
        return self.graph_retrieval.adjacency_matrix_retrieval(conn_only=conn_only)

    ################################################################

    ######## Assertions ########
    # Already fitted the graph to the data flag check
    def is_fitted(self):
        return self._fit

    # Fitting data assertion
    def _assert_input_data(self, X: np.ndarray) -> np.ndarray:
        '''
        Performs input data assertions and casting to project-standarized formats.
        Asserts the input data that the graph is to be fitted on several levels and 
        raises appropriate errors in case any of the conditions is not met. The conditions are:

            1. If the input data can be casted to np.ndarray
            2. If the input dtype can be casted to np.float32
            3. If the input data contains dupliactes (rowwise)
            4. If the input data contains missing values
            5. If the input data has consistent number of columns (dimensions) across datapoints
            tbd...

        Parameters
        ----------
        X: array-like
            Input data to be preprocessed and validated.
        
        Returns
        ----------
        np.ndarray(dtype=np.float32)
            (M x N) datapoints array of project-consistent format and shape.
        '''
        # Tylko datatype assertion jest na zewnatrz bo gdzie indziej mozemy chciec innego datatype
        # to mozna uzyc wtedy assertion wiedzac ze mamy juz np.ndarray :)
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X, dtype=np.float32)
            except:
                raise ValueError('Input datatype not recognised or cannot be casted to np.ndarray!')
        return _preprocess_ndarray_matrix(matrix=X, dtype=np.float32)

    def _validate_retrieval_fitness(self, retrieval: str):
        ''' 
        Validates if the graph has been fitted before calling the retrieval functions
        If not, raises an error and redirects the user to either fitting the data or using the generate_<RETRIVEAL> method
        '''
        if not self.is_fitted():
            raise AssertionError(self.graph_fittnes_validation_string + retrieval)

    ########################################################

    ################ Drawings ################
    @staticmethod
    def draw_data_graph(X: np.ndarray, G: nx.Graph,
                        title: str = "Grafowa reprezentacja danych", labels: bool = False, grid: bool = False, save: bool = False):
        '''
        Draws a graph on top of a 2-dimensional dataset.
        
        Parameters
        ----------
        X: np.ndarray
            Datapoints array of shape (N x 2)
        G: nx.Graph
            Graph to be drawn on top of the data
        title: str
            Plot title
        labels: bool
            Determines if the vertices of the graph should be labeled (with its indices)
        grid: bool
            Deterimnes if a grid will be shown on the graph
        save: bool
            If the graph is to be saved to file
        '''
        X = _preprocess_ndarray_matrix(X)
        data_dimension = X.shape[1]
        if data_dimension not in  (2, 3):
            raise ValueError('Only 2 and 3-dimensional data visualisation supported!')

        # Position dictionary for networkx pass
        pos = {i: [
            X[i, j] for j in range(data_dimension) # Vertix dimension depends on data dimension (2/3)
            ] 
            for i in range(X.shape[0])}

        # Initialize the graph
        fig = plt.figure(figsize=(9,9))

        # Projection of the graph
        if data_dimension == 2:
            projection = None # Default of matplotlib, 2d projection
            linewidth = 2 # Linewidth of graph edges
            edge_alpha = .8 # Alpha of graph edges
        else:
            projection = "3d"
            linewidth = 1
            edge_alpha = .5


        ax = fig.add_subplot(111, projection=projection)
        # GRID
        if grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        else:
            ax.grid(False)

        # plot vertices as scatterplot to make them align correctly in the plot
        ax.scatter(X[:, 0], X[:, 1], color='red', s=100, alpha=0)

        # Drawing nodes, edges and labels separately
        # If done otherwise nx will override the plot, axes and grid will not be displayed
        nx.draw_networkx_nodes(G, pos, node_color='#245226', edgecolors='#000000', node_size=30, alpha=0.8)

        # EDGES
        # Done obscurely for the same reason
        for u, v in G.edges():
            plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='darkblue', linewidth=linewidth)

        # LABELS
        if labels:
            if data_dimension == 2: # Labeling doesn't work (and shouldn't) for 3d data
                nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            else:
                print('Warning: Labeling the data is not and will not be supported for 3-dimensional data.')

        # TITLES
        
        plt.title(title)
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        if data_dimension == 3:
            ax.set_zlabel('Z')
        for l in fig.gca().lines:
            l.set_alpha(edge_alpha)
        
        fig.tight_layout()
        if save:
            plt_id = uuid.uuid4()
            plt.savefig(f"figures/input_{plt_id}.svg", format="svg")
            print(f'Saved as ./figures.input_{plt_id}.svg')
        plt.show()
    ################################################################


class KNNDataGraph(DataGraph):
    '''
    KNN Graph Class for KNN search over a dataset points' space.

    Attributes
    ----------

    Internal Attributes
    ----------

    Methods
    ----------

    '''

    def __init__(self,
                 k: int):  # k is considered to be heruistic-defining, so it is an attribute of the class rather than a fit() method parameter
        '''
        Initialize an instance of ```KNNDataGraph```, a class for KNN search over a dataset points' space.

        Parameters
        ----------
        k: int
            Number of nearest-neighbors to be connected in the future-calculated graphs.
        '''
        # Neighborhood size, must be int
        if type(k) != int:
            raise TypeError('Neighborhood parameter "k" must be int!')
        self.k = k
        super().__init__()

    def _generate_graph_source(self, X: np.ndarray, p: int = 2):
        '''
        Performs a graph search and generates a ```graph_source``` required for
        future retrieval of the calculated graphs. 

        Parameters
        ----------
        X: np.ndarray
            Data the graph will be fitted in. 
        p: int
            Which Minkowski p-norm to use. A large, finite p may cause a ValueError if overflow can occur.
        '''
        self._n_samples = X.shape[0]

        # KDTree for superfast lookup of knn problem
        self._kd_tree = KDTree(X)

        # KNN search
        self.weights_sparse, self.connections_sparse = self._kd_tree.query(X, k=self.k + 1,
                                                                           workers=-1)  # +1 because the search includes its own point
        self.weights_sparse = self.weights_sparse[:, 1:]
        self.connections_sparse = self.connections_sparse[:, 1:]
        # distances and indices will later allow us to transform the data into different representations of the graph


class MSTDataGraph(DataGraph):
    '''
    MST Graph Class for KNN search over a dataset points' space.

    Attributes
    ----------

    Internal Attributes
    ----------

    Methods
    ----------

    '''

    def __init__(self):
        '''
        Initialize an instance of ```MSTDataGraph```, a class for MST search over a dataset points' space.
        '''
        super().__init__()

    def fit(self, X: np.ndarray, metric: str = 'euclidean', algorithm: str = 'Prim'):
        '''
        Fits a MST graph to the input data X. 

        Parameters
        ----------
        X: np.ndarray
            Data the graph will be fitted in. 
        metric: str
            String definition of the distance metric, defaults to euclidean. Possible functions are defined
            under https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html. Defaults to ```euclidean```.
        algorithm: str
            String definition of the algorithm used. Possible algorithms are 'Prim' and 'Kruskal'. Defaults to 'Prim'.
        '''
        super().fit(X=X, metric=metric, algorithm=algorithm)

    def _generate_graph_source(self, X: np.ndarray, metric: str = 'euclidean', algorithm: str = 'Prim'):
        '''
        Performs a graph search and generates a ```graph_source``` required for
        future retrieval of the calculated graphs. 

        Parameters
        ----------
        X: np.ndarray
            Data the graph will be fitted in. 
        metric: str
            String definition of the distance metric, defaults to euclidean. Possible functions are defined
            under https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html. Defaults to ```euclidean```.
        algorithm: str
            String definition of the algorithm used. Possible algorithms are 'Prim' and 'Kruskal'. Defaults to 'Prim'.
        '''
        self._n_samples = X.shape[0]
        if algorithm == 'Prim':
            self.connections_sparse, self.weights_sparse = MSTDataGraph.prim_mst(X, metric)

        elif algorithm == 'Kruskal':  # To be implemented
            pass

            # Chyba bardzo to wolne ale dziala

    @staticmethod
    def prim_mst(X: np.ndarray, metric: str = 'euclidean') -> tuple:
        '''
        Performs Prim's algorithm on an array of datapoints, in the process effectively creating a complete
        weighted graph with weights being the value of ```metric``` parameter in the search space.

        Parameters
        ----------
        X: np.ndarray
            Array of datapoints in a continuous or discrete space.
        metric: str
            string definition of the distance metric, defaults to euclidean. Possible functions are defined
            under https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        Returns
        ----------
        tuple
            2-tuple containing connections and weights matrix. The matrices are in the format considered as
            ```graph source``` in the project.
        '''
        n = X.shape[0]

        # Pairwise distances
        distances = pdist(X, metric=metric)
        distance_matrix = squareform(distances)

        # Structures of Prim
        mst_connections = [[] for _ in range(n)]
        mst_weights = [[] for _ in range(n)]
        visited = np.zeros(n, dtype=bool)  # zeros into flags (False)
        min_heap = [(0, 0, -1)]  # (weight, current_node, parent_node)

        while min_heap:
            weight, u, parent = heappop(min_heap)  # always the smallest weight

            if visited[u]:
                continue

            visited[u] = True

            if parent != -1:  # All cases outside of the first one
                mst_connections[u].append(parent)
                mst_connections[parent].append(u)
                mst_weights[u].append(weight)
                mst_weights[parent].append(weight)

            for v in range(n):
                if not visited[v]:
                    # pushing to the min_heap, most time consuming
                    heappush(min_heap, (distance_matrix[u, v], v, u))
        return mst_connections, mst_weights

    # Jakby sie chcialo dodac jakakolwiek inną metode narysowania grafu na danych wystarczy nadpisac
    # metode _generate_graph_source, ktora ustawi poprawnie parametry:
    # self.connections_sparse,
    # self.weights_sparse

    # Reszta jest zupelnie uogolniona
    # https://www.freecodecamp.org/news/prims-algorithm-explained-with-pseudocode/
