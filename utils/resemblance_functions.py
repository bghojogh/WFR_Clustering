import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Optional, Literal
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
    cosine_similarity,
)
KernelType = Literal["cosine", "linear", "poly", "rbf", "sigmoid"]


def cosine_resemblance(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fast cosine similarity using matrix multiplication.
    """
    # cast to float:
    X = X.astype(float)
    if X2 is not None:
        X2 = X2.astype(float)

    # normalize:
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if X2 is not None:
        X2_norm = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-12)

    # resemblance:
    if X2 is not None:
        R_resemblance = X_norm @ X2_norm.T
    else:
        R_resemblance = X_norm @ X_norm.T
    return R_resemblance


def kernel_resemblance(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    kernel: KernelType = "cosine",
    **kernel_kwargs,
) -> np.ndarray:
    """
    Compute resemblance (similarity) matrix using sklearn kernel methods.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
    X2 : Optional[np.ndarray]
        Shape (m_samples, n_features). If None, uses X.
    kernel : str
        One of {"cosine", "linear", "poly", "rbf", "sigmoid"}
    kernel_kwargs :
        Extra arguments passed to the kernel function
        (e.g., gamma, degree, coef0)

    Returns
    -------
    np.ndarray
        Resemblance (kernel) matrix
    """
    # cast to float
    X = X.astype(float)
    if X2 is not None:
        X2 = X2.astype(float)

    if kernel == "cosine":
        return cosine_similarity(X, X2)

    elif kernel == "linear":
        return linear_kernel(X, X2)

    elif kernel == "poly":
        return polynomial_kernel(X, X2, **kernel_kwargs)

    elif kernel == "rbf":
        return rbf_kernel(X, X2, **kernel_kwargs)

    elif kernel == "sigmoid":
        return sigmoid_kernel(X, X2, **kernel_kwargs)

    else:
        raise ValueError(f"Unknown kernel type: {kernel}")


def kernel_cosine_resemblance(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    return kernel_resemblance(X=X, X2=X2, kernel="cosine")
def kernel_linear_resemblance(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    return kernel_resemblance(X=X, X2=X2, kernel="linear")
def kernel_poly_resemblance(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    return kernel_resemblance(X=X, X2=X2, kernel="poly")
def kernel_rbf_resemblance(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    return kernel_resemblance(X=X, X2=X2, kernel="rbf")
def kernel_sigmoid_resemblance(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    return kernel_resemblance(X=X, X2=X2, kernel="sigmoid")


def compute_resemblance_by_knn_sklearn(X_train: np.ndarray, X_test: Optional[np.ndarray] = None, 
                                       knn_k: Optional[int] = None,
                                       knn_sklearn_algorithm: Optional[str] = "auto", 
                                       resemblance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    n_train, d_train = X_train.shape
    if knn_k is None:
        knn_k = n_train
    nbrs = NearestNeighbors(n_neighbors=min(knn_k, n_train), algorithm=knn_sklearn_algorithm).fit(X_train)
    
    if X_test is None:
        distances, indices = nbrs.kneighbors(X_train)
        # compute resemblance only for neighbors
        R = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in indices[i]:
                R[i, j] = resemblance_fn(X_train[i:i+1], X_train[j:j+1])[0,0]

    else:
        n_test, d_test = X_test.shape
        distances, indices = nbrs.kneighbors(X_test)
        # compute resemblance only for neighbors
        R = np.zeros((n_test, n_train))
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                R[i, j] = resemblance_fn(X_test[i:i+1], X_train[j:j+1])[0, 0]

    return R


def compute_resemblance_by_knn_faiss(X_train: np.ndarray, X_test: Optional[np.ndarray] = None, 
                                       knn_k: Optional[int] = None,
                                       resemblance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    import faiss   # "pip install faiss-cpu" OR "pip install faiss-gpu-cu11"

    if resemblance_fn.__name__ != "cosine_resemblance":
        raise AssertionError("The FAISS approximation method for KNN can be used only when the resemblance function is cosine!")
    
    n_train, d_train = X_train.shape
    if knn_k is None:
        knn_k = n_train

    X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)  # Normalize X to unit vectors for cosine similarity
    index = faiss.IndexFlatIP(d_train)
    index.add(X_train_norm.astype(np.float32))

    if X_test is None:
        D, I = index.search(X_train_norm.astype(np.float32), knn_k)  # D = cosine similarities
        R = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j_idx, sim in zip(I[i], D[i]):
                R[i, j_idx] = sim

    else:
        n_test, d_test = X_test.shape
        X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)  # Normalize X to unit vectors for cosine similarity
        D, I = index.search(X_test_norm.astype(np.float32), knn_k)  # D = cosine similarities
        R = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j_idx, sim in zip(I[i], D[i]):
                R[i, j_idx] = sim

    return R


def compute_resemblance_by_knn_hnsw(X_train: np.ndarray, X_test: Optional[np.ndarray] = None, 
                                       knn_k: Optional[int] = None,
                                       resemblance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    import hnswlib    # pip install hnswlib
    
    if resemblance_fn.__name__ != "cosine_resemblance":
        raise AssertionError("The HNSW approximation method for KNN can be used only when the resemblance function is cosine!")
    
    n_train, d_train = X_train.shape
    if knn_k is None:
        knn_k = n_train

    # Build HNSW index
    p = hnswlib.Index(space='cosine', dim=d_train)
    p.init_index(max_elements=n_train, ef_construction=200, M=16)
    p.add_items(X_train.astype(np.float32))
    p.set_ef(knn_k * 2)  # search depth

    if X_test is None:
        I, D = p.knn_query(X_train.astype(np.float32), k=knn_k)
        R = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j_idx, dist in zip(I[i], D[i]):
                R[i, j_idx] = 1 - dist  # convert cosine distance -> similarity

    else:
        n_test, d_test = X_test.shape
        I, D = p.knn_query(X_test.astype(np.float32), k=knn_k)
        R = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j_idx, dist in zip(I[i], D[i]):
                R[i, j_idx] = 1 - dist  # convert cosine distance -> similarity

    return R