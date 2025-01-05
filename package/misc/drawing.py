from ..algorithms.algorithm_input import _AlgorithmInput
from ..graph.graph import DataGraph

import matplotlib.pyplot as plt
import uuid
import networkx as nx
import numpy as np


def draw_data(X: _AlgorithmInput,
              labels: np.array,
              title: str,
              figsize: tuple = (9, 9),
              alpha: float = 0.2,
              save=False,
              save_path: str = ' ./figures'):
    """
    Draw labeled and unlabeled data points in 2D or 3D space.

    This function visualizes data points and highlights the labeled points
    on a 2D or 3D scatter plot. It supports datasets with 2 or 3 dimensions.

    Parameters:
    -----------
    X : _AlgorithmInput
        An object containing the dataset to be visualized.

    labels : np.array
        An array of labels corresponding to the data points in `X.data`.
        The length of this array should match the number of samples in `X.data`.

    title : str
        The title of the plot.

    figsize : tuple, optional, default=(9, 9)
        The size of the figure (width, height) in inches.

    alpha : float, optional, default=0.2
        The transparency level for the unlabeled data points.

    save : bool, optional, default=False
        If True, the plot will be saved to the specified `save_path`.

    save_path : str, optional, default='./figures'
        The directory where the plot will be saved if `save` is True.

    Raises:
    -------
    ValueError
        If the dimensionality of the data exceeds 3.
    """

    X_labeled = X.get_labeled()
    fig, ax = plt.subplots(figsize=figsize)
    if X.data.shape[1] == 2:
        ax.scatter(X.data[:, 0], X.data[:, 1], c=labels, cmap='jet', alpha=alpha)
        ax.scatter(X_labeled[:, 0], X_labeled[:, 1], marker="o", c='w', s=55, edgecolors='black', linewidths=3,
                   label="Punkty sklasyfikowane")
    elif X.data.shape[1] == 3:
        ax.set_axis_off()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X.data[:, 0], X.data[:, 1], X.data[:, 2], c=labels, alpha=alpha, cmap='jet')
        ax.scatter(X_labeled[:, 0], X_labeled[:, 1], X_labeled[:, 2], marker="o", c='w', s=55, edgecolors='black',
                   linewidths=3, label="Punkty sklasyfikowane", alpha=1)
    else:
        raise ValueError('Drawing not maintained for data dimensions greater than 3!')

    ax.legend(loc="upper right",
              bbox_to_anchor=(1.01, 1.01),
              framealpha=1)

    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    if X.data.shape[1] == 3:
        ax.set_zlabel('Z')
    plt.title(title)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    plt.tight_layout()
    if save:
        plt_id = uuid.uuid4()
        plt.savefig(f"{save_path}/input_{plt_id}.svg", format="svg")
        print(f'Saved as input_{plt_id}.svg to ')
    plt.show()


def draw_graph(X: _AlgorithmInput,
               connections: np.ndarray,
               title: str,
               save: bool = False):
    """
    Draw a graph representation of data points and their connections.

    This function visualizes a graph structure where data points are represented as nodes,
    and their relationships are represented as edges based on the provided connections.

    Parameters:
    -----------
    X : _AlgorithmInput
        An object containing the dataset to be visualized.

    connections : np.ndarray
        A 2D array where each row corresponds to a node, and the elements in the row
        specify the indices of the neighboring nodes to which it is connected.

    title : str
        The title of the graph visualization.

    save : bool, optional, default=False
        If True, the graph visualization will be saved using the `DataGraph.draw_data_graph` method.
    """

    nx_dummy = nx.Graph()
    for node, neighbors in enumerate(connections):
        for neighbor in neighbors:
            nx_dummy.add_edge(node, neighbor)
    DataGraph.draw_data_graph(X.data, nx_dummy, title=title, save=save)
