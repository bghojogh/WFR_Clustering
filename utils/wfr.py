import resemblance_functions
import numpy as np
from collections import deque
from typing import Callable, Optional


class WFR(object):
    
    def __init__(
            self, 
            resemblance_threshold: float, 
            resemblance_measure: Optional[str] = "cosine",
            mark_outliers_as_minus_one: Optional[bool] = False,
            knn_k: Optional[int] = None,
            knn_sklearn_algorithm: Optional[str] = "auto",
            knn_approx_method: Optional[str] = None,
        ):
        self.resemblance_threshold = resemblance_threshold
        self.resemblance_measure = resemblance_measure
        self.mark_outliers_as_minus_one = mark_outliers_as_minus_one
        self.knn_k = knn_k
        self.knn_sklearn_algorithm = knn_sklearn_algorithm
        self.knn_approx_method = knn_approx_method
        self.X_ = None
        self.labels_ = None
        self.is_fitted_ = False
        self.R_matrix_ = None
        self.R_min_ = None
        self.R_max_ = None


    def fit(self, X: np.ndarray) -> object:
        # do clustering:
        R, labels, _ =self.compute_resemblance_and_clusters(
            X=X, 
            resemblance_fn=self.get_resemblance_function(), 
            resemblance_threshold=self.resemblance_threshold,
            mark_outliers_as_minus_one=self.mark_outliers_as_minus_one,
            knn_k=self.knn_k,
            knn_sklearn_algorithm=self.knn_sklearn_algorithm,
            knn_approx_method=self.knn_approx_method
        )
        self.X_ = X
        self.labels_ = labels
        self.is_fitted_ = True
        self.R_matrix_ = R
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

        # Full graph approach:
        if self.knn_k is None:
            R_new = resemblance_fn(X_test, X_train)

        # KNN approach:
        else:
            # use KNN graph
            match self.knn_approx_method:
                case None:
                    R_new = resemblance_functions.compute_resemblance_by_knn_sklearn(X_train=X, X_test=X_test, knn_k=knn_k, knn_sklearn_algorithm=self.knn_sklearn_algorithm, resemblance_fn=resemblance_fn)
                case "faiss":
                    R_new = resemblance_functions.compute_resemblance_by_knn_faiss(X_train=X, X_test=X_test, knn_k=knn_k, resemblance_fn=resemblance_fn)
                case "hnsw":
                    R_new = resemblance_functions.compute_resemblance_by_knn_hnsw(X_train=X, X_test=X_test, knn_k=knn_k, resemblance_fn=resemblance_fn)

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
        resemblance_threshold: Optional[float] = 0.5,
        mark_outliers_as_minus_one: Optional[bool] = False,
        knn_k: Optional[int] = None,
        knn_sklearn_algorithm: Optional[str] = "auto",
        knn_approx_method: Optional[str] = None,
        search_algorithm: Optional[str] = "DFS",
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
            mark_outliers_as_minus_one: bool
                Whether to mark outliers as -1 or not (as singleton clusters).
            knn_k: int, optional
                If given, use KNN graph with this many neighbors
            knn_sklearn_algorithm:
                algorithm used in KNN of sklearn: {'auto', 'ball_tree', 'kd_tree', 'brute'}
            knn_approx_method: str
                Options are None (for no approximation), "faiss" (Facebook AI Similarity Search), "hnsw" (Hierarchical Navigable Small World)
                Note that FAISS and HNSW are useful especially for large n (sample size) and high d (dimensionality), giving sublinear search time. 
                It can be used only when resemblance_fn is cosine_resemblance.
            search_algorithm: str
                Options are "DFS" (for depth-first search) and "BFS" (for breadth-first search). It does not have impact on the algorithm.

        Returns:
            resemblance_matrix: np.ndarray
                The (n, n) resemblance matrix
            labels: List[int]
                List of labels of the n points
            clusters: List[List[int]]
                List of clusters, each cluster is a list of point indices
        """
        if resemblance_fn is None:
            resemblance_fn = resemblance_functions.cosine_resemblance

        # Full graph approach:
        if knn_k is None:
            # Compute resemblance matrix (n x n)
            R = resemblance_fn(X)
        
        # KNN approach:
        else:
            # use KNN graph
            match knn_approx_method:
                case None:
                    R = resemblance_functions.compute_resemblance_by_knn_sklearn(X_train=X, knn_k=knn_k, knn_sklearn_algorithm=knn_sklearn_algorithm, resemblance_fn=resemblance_fn)
                case "faiss":
                    R = resemblance_functions.compute_resemblance_by_knn_faiss(X_train=X, knn_k=knn_k, resemblance_fn=resemblance_fn)
                case "hnsw":
                    R = resemblance_functions.compute_resemblance_by_knn_hnsw(X_train=X, knn_k=knn_k, resemblance_fn=resemblance_fn)

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

        # Ensure self-loops for DFS or BFS
        np.fill_diagonal(adjacency, True)

        # Find connected components (DFS or BFS)
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        clusters = []
        for i in range(n):
            if not visited[i]:
                if search_algorithm == "DFS":
                    # it is a stack
                    stack_or_queue = [i]
                elif search_algorithm == "BFS":
                    # it is a queue
                    stack_or_queue = deque([i])
                cluster = []

                while stack_or_queue:
                    if search_algorithm == "DFS":
                        node = stack_or_queue.pop()
                    elif search_algorithm == "BFS":
                        node = stack_or_queue.popleft()
                    if visited[node]:
                        continue
                    visited[node] = True
                    labels[node] = cluster_id
                    cluster.append(node)

                    neighbors = np.where(adjacency[node])[0]
                    for nb in neighbors:
                        if not visited[nb]:
                            stack_or_queue.append(nb)

                clusters.append(cluster)
                cluster_id += 1

        # optionally mark singleton clusters as -1
        if mark_outliers_as_minus_one:
            for cluster in clusters:
                if len(cluster) == 1:
                    labels[cluster[0]] = -1

        return R, labels, clusters


# test:
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(10, 5)

    resemblance_threshold = 0.8
    mark_outliers_as_minus_one = False
    knn_k = None

    wfr = WFR(resemblance_threshold=resemblance_threshold, resemblance_measure="cosine", mark_outliers_as_minus_one=mark_outliers_as_minus_one, knn_k=knn_k)
    labels = wfr.fit_predict(X=X)
    print(labels)

    labels = wfr.predict(X=X)
    print(labels)