from graph.graph import MSTDataGraph
from algorithms.algorithm_input import _AlgorithmInput
import numpy as np
from collections import defaultdict


def make_mst(X: _AlgorithmInput):
    '''
    Performs a graph search and generates a ```graph_source``` required for
    future retrieval of the calculated graphs.

    Parameters
    ----------
    X: np.ndarray
        Data the graph will be fitted in.
    '''
    mst = MSTDataGraph()
    mst.fit(X.data)
    weights_sparse = mst.weights_sparse
    connections_sparse = mst.connections_sparse

    return connections_sparse, weights_sparse


def find_all_paths_with_labels(connections_sparse: np.ndarray, labels: np.array, start: int, bomb_node: int = None,
                               path=None):
    """
    Find all paths from the starting node to nodes with labels greater than -1 using DFS algorithm.
    Stop searching a path if a node with the same label as the starting node is encountered,
    if a node with a label > -1 is found, or if a "bomb node" is encountered.

    Parameters
    ----------
    connections_sparse: np.ndarray
        Object that for each index of the first dimension has a list representation of vertices connected to the index.
    labels: np.array
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
    if path is None:
        path = []

    # Extend the current path
    path = path + [start]

    # Initialize a list to collect all valid paths
    all_paths = []

    # Check if the current node is the "bomb node"
    if bomb_node is not None and start == bomb_node:
        return []  # Stop and discard this path if bomb node is reached

    # Check if the current node satisfies the condition
    if labels[start] > -1 and start != path[0]:  # Stop and return path if label > -1
        return [path]
    # Explore neighbors
    for neighbor in connections_sparse[start]:
        if neighbor not in path:
            # Stop exploring if the neighbor's label matches the start node's label
            if labels[neighbor] == labels[path[0]]:
                continue

            # Recursively find paths from the neighbor
            new_paths = find_all_paths_with_labels(
                connections_sparse=connections_sparse,
                labels=labels,
                start=neighbor,
                bomb_node=bomb_node,
                path=path
            )
            # Append all found paths
            all_paths.extend(new_paths)

    return all_paths


def explore_graph_recursive(connections_sparse: np.ndarray, labels: np.array, start: int):
    """
    Explore the graph recursively by finding all paths from a starting node.
    After finding paths, recursively call the function for the end nodes of each path,
    passing the current start as a bomb node to prevent backtracking.

    Parameters
    ----------
    connections_sparse: np.ndarray
        Graph adjacency list in sparse format.
    labels: np.array
        Array of labels for each node.
    start: int
        Starting node.

    Returns
    ----------
    List of all paths explored during the traversal.
    """

    def recursive_explore(current_start, bomb_node, visited_paths):
        # Find all paths from the current node
        paths = find_all_paths_with_labels(connections_sparse, labels, current_start, bomb_node=bomb_node)


        # Add all found paths to the visited list
        visited_paths.extend(paths)

        # For each path's endpoint, continue exploration
        for path in paths:
            end_node = path[-1]
            if end_node != bomb_node:# Avoid backtracking
                recursive_explore(current_start=end_node, bomb_node=path[-2], visited_paths=visited_paths)

    # Initialize list to store all visited paths
    all_paths = []

    recursive_explore(current_start=start, bomb_node=None, visited_paths=all_paths)
    return all_paths

def find_subgraph_with_labels(connections_sparse, weights_sparse, labels):
    """
    Creates subgraph of original graph that consists of paths between nodes with different labels. If node is not
    labeled it is not treated as a obstacle for path
    Parameters
    ----------
    labels: np.array Array in which for
        each index that refers to node there is label information.

    Returns
    -------
    sub_connections_sparse, sub_weights_sparse: np.darray
        Objects similar to connections_sparse and weights_sparse that define a subgraph of original graph

    """

    all_paths = []

    node_list = [i for i in range(len(labels)) if labels[i] != -1]
    unvisited_labeled_nodes = node_list.copy()
    nodes_visited = set()
    while unvisited_labeled_nodes:
        start_node = unvisited_labeled_nodes.pop(0)
        new_paths = explore_graph_recursive(connections_sparse, labels, start_node)

        if len(new_paths) > 0:
            for path in new_paths:
                all_paths.append(path)
            for path in all_paths:
                nodes_visited.update(path)
        else:
            nodes_visited.add(start_node)
        unvisited_labeled_nodes = [node for node in node_list if node not in nodes_visited]



    sub_connections_sparse = [[] for _ in range(len(connections_sparse))]
    sub_weights_sparse = [[] for _ in range(len(connections_sparse))]

    for path in all_paths:
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]

            if node_b not in sub_connections_sparse[node_a]:
                sub_connections_sparse[node_a].append(node_b)

                edge_index = connections_sparse[node_a].index(node_b)
                edge_value = weights_sparse[node_a][edge_index]
                sub_weights_sparse[node_a].append(edge_value)

            if node_a not in sub_connections_sparse[node_b]:
                sub_connections_sparse[node_b].append(node_a)

    return sub_connections_sparse, sub_weights_sparse


def find_and_pop_max_edge(weights_sparse, connections_sparse, sub_weights_sparse: list):
    """
    Chooses max edge from subgraph and pops it from original graph.
    Parameters
    ----------
    sub_weights_sparse: np.ndarray
        Object that for each index of the first dimension has a list representation edges' weights coming out of the
        index. It is a product of find_subgraph_with_labels function.
    -------
    """

    max_distance = float('-inf')
    node_label1 = -1

    for i, inner_list in enumerate(sub_weights_sparse):
        for j, value in enumerate(inner_list):
            if value > max_distance:
                max_distance = value
                node_label1 = i

    edge_index = weights_sparse[node_label1].index(max_distance)
    node_label2 = connections_sparse[node_label1][edge_index]
    edge_index2 = weights_sparse[node_label2].index(max_distance)

    connections_sparse[node_label1].pop(edge_index)
    connections_sparse[node_label2].pop(edge_index2)
    weights_sparse[node_label1].pop(edge_index)
    weights_sparse[node_label2].pop(edge_index2)

    return weights_sparse, connections_sparse


def make_labels(node, connections_sparse, label, visited, labels):
    """
    Gives the same label to nodes that are connected with each-other. Makes modification to existing labels array.
    Parameters
    ----------
    node: int
        Index of starting node
    label: int
        Label of starting node
    visited:
        Set of visited nodes
    labels: np.array
        Array in which for each index that refers to node there is label information.
    -------
    """

    visited.add(node)
    labels[node] = label

    for neighbor in connections_sparse[node]:
        if neighbor not in visited:
            make_labels(neighbor, connections_sparse, label, visited, labels)




def clusterize(weights_sparse, connections_sparse, labels):
    """
    Main function for clusterization the data with given labels. The assumption is that every label has at least
    one representative in data, and it is marked correctly. The algorythm is a modification of traditional edge
    cutting in mst graph. Instead of cutting the longest k-1 edges where k is the expected number of clusters it
    find subgraph of edges that are part of path that connects nodes of different labels. It chooses edges from
    that subgraph and cuts them until all nodes with different labels are separated. At the end it checks if
    nodes with the same labels are connected and connects them if they are not.
    Parameters
    ----------
    labels: np.array
        Array in which for each index that refers to node there is label information.
    Returns
    -------
    new_labels: np.array
        Array of new labels. In this array all nodes are assigned to a label.

    """
    i = 0
    while True:
        sub_connections_sparse, sub_weights_sparse = find_subgraph_with_labels(weights_sparse, connections_sparse, labels)
        i += 1
        if all(len(sublist) == 0 for sublist in sub_connections_sparse):
            break
        weights_sparse, connections_sparse = find_and_pop_max_edge(weights_sparse, connections_sparse, sub_weights_sparse)



    labels_groups = defaultdict(list)
    for index, value in enumerate(labels):
        if value != -1:
            labels_groups[value].append(index)

    labels_groups = [indices for indices in labels_groups.values()]
    same_labels_groups = [indices for indices in labels_groups if len(indices) > 1]

    if not all(len(sublist) == 0 for sublist in same_labels_groups):
        for label_group in same_labels_groups:
            for i in range(len(label_group) - 1):
                start_node = label_group[i]
                end_node = label_group[i + 1]
                connections_sparse[start_node].append(end_node)
                connections_sparse[end_node].append(start_node)
                weights_sparse[start_node].append(1)
                weights_sparse[end_node].append(1)

    cluster_representative_nodes = [group[0] for group in labels_groups]
    new_labels = labels.copy()
    for node in cluster_representative_nodes:
        make_labels(node, connections_sparse, new_labels[node], visited=set(), labels=new_labels)

    return weights_sparse, connections_sparse, new_labels