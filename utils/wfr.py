from utils import resemblance_functions, clustering_scores
import numpy as np
from collections import deque
from tqdm import tqdm
from typing import Callable, Optional, Tuple, List


class WFR(object):
    
    def __init__(
            self, 
            resemblance_measure: Optional[str] = "log_based",
            resemblance_threshold: Optional[float] = None,
            resemblance_threshold_grid_search_step: Optional[float] = 0.01,
            mark_outliers_as_minus_one: Optional[bool] = False,
            knn_k: Optional[int] = 10,
            knn_sklearn_algorithm: Optional[str] = "auto",
            knn_approx_method: Optional[str] = None,
            kernel: Optional[str] = "rbf"
        ) -> None:
        """
        Args:
            resemblance_measure: str, optional
                The measure for calculating resemblance of data points.
                Defaults to cosine similarity.
                Options: log_based, cosine, kernel
            resemblance_threshold: float, optional
                Value in [0, 1] for thresholding normalized resemblance matrix.
                If not provided, automatic threshold is used, but that will slow down the clustering. 
            resemblance_threshold_grid_search_step: float, optional
                Value in [0, 1] for the step of grid search for the best resemblance_threshold.
                This is used only when resemblance_threshold is None (not provided).
            mark_outliers_as_minus_one: bool
                Whether to mark outliers as -1 or not (as singleton clusters).
            knn_k: int, optional
                Use KNN graph with this many neighbors.
                If given as None, all points are used, i.e., k=n 
            knn_sklearn_algorithm:
                algorithm used in KNN of sklearn: {'auto', 'ball_tree', 'kd_tree', 'brute'}
            knn_approx_method: str
                Options are None (for no approximation), "faiss" (Facebook AI Similarity Search), "hnsw" (Hierarchical Navigable Small World)
                Note that FAISS and HNSW are useful especially for large n (sample size) and high d (dimensionality), giving sublinear search time. 
                It can be used only when resemblance_fn is cosine_resemblance.
            kernel: str
                Options are cosine, linear, poly, rbf, sigmoid
                Used only when resemblance_measure="kernel"
        """
        self.resemblance_measure = resemblance_measure
        self.resemblance_threshold = resemblance_threshold
        self.resemblance_threshold_grid_search_step = resemblance_threshold_grid_search_step
        self.mark_outliers_as_minus_one = mark_outliers_as_minus_one
        self.knn_k = knn_k
        self.knn_sklearn_algorithm = knn_sklearn_algorithm
        self.knn_approx_method = knn_approx_method
        self.kernel = kernel
        self.X_ = None
        self.X_mean_ = None
        self.labels_ = None
        self.is_fitted_ = False
        self.R_matrix_ = None
        self.R_min_ = None
        self.R_max_ = None
        self.adjacency_ = None
        self.adjacency_test_ = None


    def fit(self, X: np.ndarray) -> object:
        # center data if necessary:
        if self.resemblance_measure == "cosine":
            X_mean = np.mean(X, axis=0, keepdims=True)
            self.X_mean_ = X_mean
            X = X - X_mean
        
        if self.resemblance_threshold is not None:
            R, labels, _, adjacency =self.compute_resemblance_and_clusters(X=X)
        else:
            clustering_scores_list, labels_list, R_list, adjacency_list, resemblance_threshold_list = [], [], [], [], []
            for resemblance_threshold in tqdm(np.arange(1.0, 0-self.resemblance_threshold_grid_search_step, -self.resemblance_threshold_grid_search_step), desc="Grid search for resemblance threshold"):
                self.resemblance_threshold = resemblance_threshold
                R, labels, _, adjacency =self.compute_resemblance_and_clusters(X=X)
                clustering_score1 = clustering_scores.graph_cluster_separation_score(X=X[labels!=-1, :], labels=labels[labels!=-1], k=10, eps=1e-8)
                clustering_score2 = clustering_scores.cluster_size_score(labels=labels[labels!=-1], min_frac=0.05, alpha=2.0)
                clustering_scores_list.append(clustering_score1 + clustering_score2)
                labels_list.append(labels)
                R_list.append(R)
                adjacency_list.append(adjacency)
                resemblance_threshold_list.append(resemblance_threshold)

            # choose the best clustering score:
            # print(clustering_scores_list)  # uncomment for debugging
            best_clustering_index = np.argmax(clustering_scores_list)
            labels = labels_list[best_clustering_index]
            R = R_list[best_clustering_index]
            adjacency = adjacency_list[best_clustering_index]
            self.resemblance_threshold = resemblance_threshold_list[best_clustering_index]

        self.X_ = X
        self.labels_ = labels
        self.is_fitted_ = True
        self.R_matrix_ = R
        self.R_min_ = self.R_matrix_.min()
        self.R_max_ = self.R_matrix_.max()
        self.adjacency_ = adjacency
        return self


    def fit_predict(self, X: np.ndarray) -> np.array:
        self.fit(X)
        return self.labels_


    def predict(self, X: np.ndarray) -> np.array:
        if not self.is_fitted_:
            raise ValueError("The object of WFR class is not fitted (trained) yet!")

        X_test = X.astype(float)
        X_train = self.X_.astype(float)

        # center data if necessary:
        if self.resemblance_measure == "cosine":
            # self.X_ (X_train) is already centered.
            X_test = X_test - self.X_mean_

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
                    R_new = resemblance_functions.compute_resemblance_by_knn_sklearn(X_train=X_train, X_test=X_test, knn_k=self.knn_k, knn_sklearn_algorithm=self.knn_sklearn_algorithm, resemblance_fn=resemblance_fn)
                case "faiss":
                    R_new = resemblance_functions.compute_resemblance_by_knn_faiss(X_train=X_train, X_test=X_test, knn_k=self.knn_k, resemblance_fn=resemblance_fn)
                case "hnsw":
                    R_new = resemblance_functions.compute_resemblance_by_knn_hnsw(X_train=X_train, X_test=X_test, knn_k=self.knn_k, resemblance_fn=resemblance_fn)

        # Normalize using training min/max
        R_new_norm = (R_new - self.R_min_) / (self.R_max_ - self.R_min_)
        R_new_norm = np.clip(R_new_norm, 0, 1)

        # Assign label of training point with largest resemblance above threshold
        labels_test = -np.ones(X_test.shape[0], dtype=int)
        max_indices = np.argmax(R_new_norm, axis=1)
        max_values = R_new_norm[np.arange(X_test.shape[0]), max_indices]

        # Assign label if max resemblance >= threshold
        if self.mark_outliers_as_minus_one:
            above_threshold = max_values >= self.resemblance_threshold
        else:
            above_threshold = np.ones_like(max_values, dtype=bool)
        labels_test[above_threshold] = self.labels_[max_indices[above_threshold]]

        # set the adjacency-test matrix:
        self.adjacency_test_ = np.zeros((X_test.shape[0], X_train.shape[0]))
        for i in range(X_test.shape[0]):
            if above_threshold[i]:
                self.adjacency_test_[i, max_indices[i]] = 1

        return labels_test


    def get_resemblance_function(self) -> Callable:
        match self.resemblance_measure:
            case "log_based":
                resemblance_fn = resemblance_functions.log_based_resemblance
            case "cosine":
                resemblance_fn = resemblance_functions.cosine_resemblance
            case "kernel":
                if self.kernel == "cosine":
                    resemblance_fn = resemblance_functions.kernel_cosine_resemblance
                elif self.kernel == "linear":
                    resemblance_fn = resemblance_functions.kernel_linear_resemblance
                elif self.kernel == "poly":
                    resemblance_fn = resemblance_functions.kernel_poly_resemblance
                elif self.kernel == "rbf":
                    resemblance_fn = resemblance_functions.kernel_rbf_resemblance
                elif self.kernel == "sigmoid":
                    resemblance_fn = resemblance_functions.kernel_sigmoid_resemblance
            case _:
                resemblance_fn = resemblance_functions.cosine_resemblance
        return resemblance_fn


    def compute_resemblance_and_clusters(
        self,
        X: np.ndarray,
        search_algorithm: Optional[str] = "DFS",
        outlier_detection_method: Optional[str] = "max_ratio"
    ) -> Tuple[np.ndarray, List[int], List[List[int]], np.ndarray[bool]]:
        """
        Args:
            X: np.ndarray
                Shape (n, d), n samples and d features
            search_algorithm: str
                Options are "DFS" (for depth-first search) and "BFS" (for breadth-first search). It does not have impact on the algorithm.
            outlier_detection_method: str
                Options are "max_ratio" and "statistical"

        Returns:
            resemblance_matrix: np.ndarray
                The (n, n) resemblance matrix
            labels: List[int]
                List of labels of the n points
            clusters: List[List[int]]
                List of clusters, each cluster is a list of point indices
            adjacency: np.ndarray[bool]
                The adjacency matrix of the data points, based on the thresholded resemblances
        """
        # get the resemblance function:
        resemblance_fn = self.get_resemblance_function()

        # Full graph approach:
        if self.knn_k is None:
            # Compute resemblance matrix (n x n)
            R = resemblance_fn(X)
        
        # KNN approach:
        else:
            # use KNN graph
            match self.knn_approx_method:
                case None:
                    R = resemblance_functions.compute_resemblance_by_knn_sklearn(X_train=X, knn_k=self.knn_k, knn_sklearn_algorithm=self.knn_sklearn_algorithm, resemblance_fn=resemblance_fn)
                case "faiss":
                    R = resemblance_functions.compute_resemblance_by_knn_faiss(X_train=X, knn_k=self.knn_k, resemblance_fn=resemblance_fn)
                case "hnsw":
                    R = resemblance_functions.compute_resemblance_by_knn_hnsw(X_train=X, knn_k=self.knn_k, resemblance_fn=resemblance_fn)

        # Normalize to [0, 1]
        r_min = R.min()
        r_max = R.max()
        if r_max > r_min:
            R_norm = (R - r_min) / (r_max - r_min)
        else:
            R_norm = np.zeros_like(R)

        # Threshold
        R_thresh = np.where(R_norm >= self.resemblance_threshold, R_norm, 0.0)

        # Build graph adjacency (boolean)
        adjacency = R_thresh > 0

        # make the adjacency matrix symmtric (it may be asymetric when k < n in KNN)
        adjacency = adjacency | adjacency.T  # ensure symmetry

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

        # optionally mark outlier points as label -1
        if self.mark_outliers_as_minus_one:
            if outlier_detection_method == "max_ratio":
                cluster_indices_to_be_removed = clustering_scores.find_very_small_max_ratio(nums=[len(cluster) for cluster in clusters], ratio=0.05)
            elif outlier_detection_method == "statistical":
                cluster_indices_to_be_removed = clustering_scores.find_very_small_statistical(nums=[len(cluster) for cluster in clusters], k=2)
            for cluster_index, cluster in enumerate(clusters):
                if cluster_index in cluster_indices_to_be_removed:
                    labels[cluster] = -1
            if -1 in labels:
                # relabel remaining clusters to start from 0
                unique_labels = sorted(l for l in set(labels) if l >= 0)
                label_mapping = {old: new for new, old in enumerate(unique_labels)}

                for old, new in label_mapping.items():
                    labels[labels == old] = new

        return R, labels, clusters, adjacency


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