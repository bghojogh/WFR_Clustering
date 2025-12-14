import resemblance_functions
import numpy as np
from typing import Callable, Optional, List


class WFR(object):
    
    def __init__(
            self, 
            resemblance_threshold: float, 
            resemblance_measure: Optional[str] = "cosine"
        ):
        self.resemblance_threshold = resemblance_threshold
        self.resemblance_measure = resemblance_measure
        self.X_ = None
        self.labels_ = None
        self.is_fitted_ = False
        self.R_matrix_ = None
        self.R_min_ = None
        self.R_max_ = None


    def fit(self, X: np.ndarray) -> object:
        # do clustering:
        R_thresh, labels, _ =self.compute_resemblance_and_clusters(
            X=X, 
            resemblance_fn=self.get_resemblance_function(), 
            resemblance_threshold=self.resemblance_threshold
        )
        self.X_ = X
        self.labels_ = labels
        self.is_fitted_ = True
        self.R_matrix_ = R_thresh
        self.R_min_ = self.R_matrix_.min()
        self.R_max_ = self.R_matrix_.max()
        return self


    def fit_predict(self, X: np.ndarray) -> np.array:
        self.fit(X)
        return self.labels_


    def predict(self, X: np.ndarray) -> np.array:
        if not self.is_fitted_:
            raise ValueError("The object of WTF class is not fitted (trained) yet!")

        X_test = X.astype(float)
        X_train = self.X_.astype(float)

        # resemblance between test points and training points
        resemblance_fn = self.get_resemblance_function()
        R_new = resemblance_fn(X_test, X_train)

        # Normalize using training min/max
        R_new_norm = (R_new - self.R_min_) / (self.R_max_ - self.R_min_)
        R_new_norm = np.clip(R_new_norm, 0, 1)

        # Assign label of training point with largest resemblance above threshold
        labels_test = -np.ones(X_test.shape[0], dtype=int)
        max_indices = np.argmax(R_new_norm, axis=1)
        max_values = R_new_norm[np.arange(X_test.shape[0]), max_indices]

        # Assign label if max resemblance >= threshold
        above_threshold = max_values >= self.resemblance_threshold
        labels_test[above_threshold] = self.labels_[max_indices[above_threshold]]

        return labels_test

    def get_resemblance_function(self) -> Callable:
        match self.resemblance_measure:
            case "cosine":
                resemblance_fn = resemblance_functions.cosine_resemblance
            case _:
                resemblance_fn = resemblance_functions.cosine_resemblance
        return resemblance_fn


    def compute_resemblance_and_clusters(
        self,
        X: np.ndarray,
        resemblance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        resemblance_threshold: float = 0.5
    ):
        """
        Args:
            X: np.ndarray
                Shape (n, d), n samples and d features
            resemblance_fn: callable, optional
                Function that takes X and returns an (n, n) resemblance matrix.
                Defaults to cosine similarity.
            resemblance_threshold: float
                Value in [0, 1] for thresholding normalized resemblance matrix.

        Returns:
            resemblance_matrix: np.ndarray
                Thresholded and normalized (n, n) resemblance matrix
            labels: List[int]
                List of labels of the n points
            clusters: List[List[int]]
                List of clusters, each cluster is a list of point indices
        """
        if resemblance_fn is None:
            resemblance_fn = resemblance_functions.cosine_resemblance

        # Compute resemblance matrix (n x n)
        R = resemblance_fn(X)

        # Normalize to [0, 1]
        r_min = R.min()
        r_max = R.max()
        if r_max > r_min:
            R_norm = (R - r_min) / (r_max - r_min)
        else:
            R_norm = np.zeros_like(R)

        # Threshold
        R_thresh = np.where(R_norm >= resemblance_threshold, R_norm, 0.0)

        # Build graph adjacency (boolean)
        adjacency = R_thresh > 0
        np.fill_diagonal(adjacency, True)

        # Find connected components (DFS / BFS)
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        clusters = []
        for i in range(n):
            if not visited[i]:
                stack = [i]
                cluster = []

                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    labels[node] = cluster_id
                    cluster.append(node)

                    neighbors = np.where(adjacency[node])[0]
                    for nb in neighbors:
                        if not visited[nb]:
                            stack.append(nb)

                clusters.append(cluster)
                cluster_id += 1

        return R_thresh, labels, clusters
    
# test:
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(10, 5)

    wfr = WFR(resemblance_threshold=0.8, resemblance_measure="cosine")
    labels = wfr.fit_predict(X=X)
    print(labels)

    labels = wfr.predict(X=X)
    print(labels)