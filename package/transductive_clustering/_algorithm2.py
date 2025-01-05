from ..utils._parameter_validation import Interval
from ._transductive_estimator import GraphBasedTransductiveEstimator
from ..algorithms.src.algorithm2.binaries.mknn import mutual_knn, informative_edges
from ..algorithms.src.algorithm2.binaries.mknn_bfs import find_connected_components
from ..algorithms.src.algorithm2.binaries.mknn_connect import connect_unlabeled
from ..algorithms.src.algorithm2.binaries.mknn_lp import weighted_label_propagation

class Algorithm2(GraphBasedTransductiveEstimator):
    _parameter_constraints: dict = {
        "k": Interval(int, 1, 1e32, closed="left")
    }
    transitional_graphs_types = ("knn", "mknn", "informative_edges")
    def __init__(self, k: int):
        super().__init__(k=k)

        self.k = k
        self.transitional_graphs_ = {k: None for k in Algorithm2.transitional_graphs_types}

    def fit(self, X, y):
        super().fit(X, y)
        self.generate_graph()
        self.labels_ = weighted_label_propagation(self.graph_connections_,
                                                  self.graph_weights_, self.y_)

        self.is_fitted = True
    def fit_predict(self, X, y):
        super().fit_predict(X, y)
        return self.labels_
    def _construct_graph(self):
        _m_c, _m_w, _kc, _kw = mutual_knn(self._algorithm_input.data, self.k)
        self.transitional_graphs_["knn"] = ({"connections": _kc,
                                             "weights": _kw})
        self.transitional_graphs_["mknn"] = ({"connections": _m_c.to_numpy_all(),
                                             "weights": _m_w.to_numpy_all()})
        
        _ie_c, _ie_w = informative_edges(self._algorithm_input,self.k,_m_c,_m_w)
        self.transitional_graphs_["informative_edges"] = ({"connections": _ie_c.to_numpy_all(),
                                                           "weights": _ie_w.to_numpy_all()})

        _comp, _vcomp = find_connected_components(self._algorithm_input, _ie_c)
        self.graph_connections_, self.graph_weights_ = connect_unlabeled(_kc, _kw, _comp, _vcomp, 
                                                                         _ie_c,
                                                                         _ie_w)