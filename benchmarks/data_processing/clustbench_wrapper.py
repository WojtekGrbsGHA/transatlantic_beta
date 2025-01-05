import clustbench
import numpy as np


def generate_clustbench_data(battery: str,
                             dataset: str,
                             p: int,  # Fraction of dataset or number of points
                             label_version: int = 0,
                             seed: int = None,
                             datapath: str = './datasets',
                             even_split: bool = True) -> tuple[np.array, np.array]:
    """
    Generate a split of labeled and unlabeled data from a specified dataset. 
    `p` [*fraction* or *amount of datapoints*] labels retain their values, 
    the rest is set to `-1`.
    The data used must come from Benchmark Suite (v1.1.0), see: https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html#sec-suite-v1
    
    Parameters
    ----------
    battery : str
        The dataset battery or collection name within ClustBench.
    
    seed : int
        Random seed for reproducibility 
    
    dataset : str
        The specific dataset name within the battery to load.

    p : float or int
        If a float between 0 and 1, `p` represents the fraction of data 
        points to include in the labeled set. If an integer greater than  1,
        it specifies the exact number of labeled samples to use. In either case, 
        at least one sample from each class is included.

    label_version : int, optional, default=0
        Specifies the label version to use if multiple labeling schemes are available.

    datapath : str, optional, default='./datasets'
        Path to the directory where datasets are stored.

    even_split : bool, optional, default=True
        Determines if the split is balanced across the distinct classess of the labeled data
    """
    if seed:
        np.random.seed(seed)
    clustbench_data = clustbench.load_dataset(battery, dataset, path=datapath)
    data = clustbench_data.data
    labels = clustbench_data.labels[label_version]

    labels = np.array(labels)
    n_samples = len(labels)

    # Determine the number of labels to keep
    if p < 1:
        n_label = int(p * n_samples)
    elif p == 1:
        n_label = n_samples
    else:
        n_label = int(p)
    n_label = min(n_label, n_samples)

    labels_modified = np.full_like(labels, -1)
    classes, counts = np.unique(labels, return_counts=True)
    assert len(classes) <= n_label, ('Number of unique labels can not be higher than the number of data '
                                           'points to include in the labeled set')
    if not even_split:
        # Firstly select one observation for every label, then randomly select (n_label - number of unique clusters)
        # indices to keep
        labeled_indices = []
        for label in classes:
            label_indices = np.where(labels == label)[0]
            labeled_indices.append(np.random.choice(label_indices))

        remaining_labels_to_select = n_label - len(labeled_indices)
        if remaining_labels_to_select > 0:
            remaining_indices = np.setdiff1d(np.arange(n_samples), labeled_indices)
            additional_indices = np.random.choice(remaining_indices, remaining_labels_to_select, replace=False)
            labeled_indices.extend(additional_indices)

        # Assign labels to the selected indices
        labeled_indices = np.array(labeled_indices)
        labels_modified[labeled_indices] = labels[labeled_indices]

    else:
        # Evenly distribute labels among classes
        n_classes = len(classes)
        per_class_labels = np.zeros(n_classes, dtype=int)

        total_labels = 0
        class_index = 0

        # Distribute labels to each class in a round-robin fashion
        while total_labels < n_label:
            if per_class_labels[class_index] < counts[class_index]:
                per_class_labels[class_index] += 1
                total_labels += 1
            class_index = (class_index + 1) % n_classes
            if np.all(per_class_labels >= counts):
                break

        # Select indices for each class
        labeled_indices = []
        for i, class_label in enumerate(classes):
            class_indices = np.where(labels == class_label)[0]
            np.random.shuffle(class_indices)
            n_select = per_class_labels[i]
            selected_indices = class_indices[:n_select]
            labeled_indices.extend(selected_indices)

        labels_modified[labeled_indices] = labels[labeled_indices]

    return data, labels_modified
