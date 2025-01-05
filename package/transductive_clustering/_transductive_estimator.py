from ..preprocess.preprocessing import _preprocess_labels, _preprocess_ndarray_matrix
from ..utils._parameter_validation import validate_parameters
from ..algorithms.algorithm_input import _AlgorithmInput
from ..misc.drawing import draw_graph
class TransductiveEstimator:

    _parameter_constraints: dict

    def _validate_parameters(cls, param_dict):
        validate_parameters(cls._parameter_constraints, param_dict)

    def __init__(self, **kwargs):
        self.input_wrapped = False
        self._graph_generated = False
        self.is_fitted = False
        self._validate_parameters(kwargs)

    def fit(self, X, y):
        self.X_ = _preprocess_ndarray_matrix(X)
        # self.y_ = _preprocess_labels(y)
        self.y_ = y
        if not self.input_wrapped:
            self._wrap_input()
    def fit_predict(self, X, y):
        if not self.is_fitted:
            self.fit(X,y)
    
    def _wrap_input(self):
        self._algorithm_input = _AlgorithmInput(self.X_, self.y_)
        self.input_wrapped = True
    
class GraphBasedTransductiveEstimator(TransductiveEstimator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X,y):
        super().fit(X,y)

    def fit_predict(self, X, y):
        super().fit_predict(X, y)

    def generate_graph(self):
        if not self.input_wrapped:
            self._wrap_input()
        self._construct_graph()
        self._graph_generated = True

    def _construct_graph():
        pass

    def draw_result_graph(self, title='', save=False):
        if not self._graph_generated:
            raise AssertionError("Graph not yet generated for this estimator instance. "+
                                 "Try using fitting the estimator or generating the graph first!")
        if self.graph_connections_ is None or self.graph_weights_ is None:
            raise AssertionError("Graph was generated for this estimator instance, but its properties "+
                                 "are missing. Try regenerating the graph or manually provide its data.")
        draw_graph(self._algorithm_input,
                   self.graph_connections_.to_numpy_all(),
                   title,
                   save)
        
    def draw_transitional_graph(self, graph_type, title='', save=False):
        if graph_type not in self.transitional_graphs_types:
            raise ValueError(f"Graph type not recognised for the algorithm. Supported graphs are: {self.transitional_graphs_types}")
        if not self._graph_generated:
            raise AssertionError("Graph not yet generated for this estimator instance. "+
                                 "Try using fitting the estimator or generating the graph first.")
        if self.graph_connections_ is None or self.graph_weights_ is None:
            raise AssertionError("Graph was generated for this estimator instance, but its properties "+
                                 "are missing. Try re-generating the graph or manually provide its data.")
        draw_graph(self._algorithm_input,
                   self.transitional_graphs_[graph_type]["connections"],
                   title,
                   save)