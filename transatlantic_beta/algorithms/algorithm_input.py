import numpy as np

class _AlgorithmInput:

    def __init__(self, X, labels, copy_dataset=False):
        self.n = len(X)
        if copy_dataset:
            self.data = X.copy()
        else:
            self.data = X
            
        self.labeled_indices = np.where(labels != -1)[0]
        self.unlabeled_indices = np.where(labels == -1)[0]

        self.labeled_count = len(self.labeled_indices)
        self.unlabeled_count = len(X) - self.labeled_count

    def get_unlabeled(self):
        return self.data[self.unlabeled_indices,:]

    def get_labeled(self):
        return self.data[self.labeled_indices,:]  

    