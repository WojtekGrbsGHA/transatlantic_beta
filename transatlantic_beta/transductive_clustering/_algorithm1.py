from ..utils._parameter_validation import Interval
from ._transductive_estimator import GraphBasedTransductiveEstimator
from ..algorithms.src.algorithm1.binaries.mstl import prim_mst, clusterize
import copy
import numpy as np

class Algorithm1(GraphBasedTransductiveEstimator):
    _parameter_constraints: dict = {}
    transitional_graphs_types = ("mst")
    def __init__(self):
        super().__init__()
        self.transitional_graphs_ = {k: None for k in Algorithm1.transitional_graphs_types}

    def fit(self, X, y):
        super().fit(X, y)
        self.generate_graph()

        self.is_fitted = True
    def fit_predict(self, X, y):
        super().fit_predict(X, y)
        return self.labels_
    def _construct_graph(self):
        connections_sparse, weights_sparse = prim_mst(self._algorithm_input.data)
        self.transitional_graphs_["mst"] = {"connections": connections_sparse,
                                            "weights": weights_sparse}
        self.graph_connections_, self.graph_weights_, self.labels_ = clusterize(connections_sparse, 
                                                                                weights_sparse,
                                                                                self.y_.astype(np.int8))