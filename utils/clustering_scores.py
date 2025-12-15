import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def graph_cluster_separation_score(X, labels, k=10, eps=1e-8):
    """
    Graph-based cluster separation score in [0, 1].

    Args:
        X: ndarray of shape (n_samples, n_features)
        labels: ndarray of shape (n_samples,)
        k: number of nearest neighbors
        eps: numerical stability constant

    Returns:
        float: separation score (1 = perfectly separated)
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    inter_weight = 0.0
    total_weight = 0.0

    for i in range(len(X)):
        for j, d in zip(indices[i][1:], distances[i][1:]):  # skip self
            w = 1.0 / (d + eps)
            total_weight += w
            if labels[i] != labels[j]:
                inter_weight += w

    return 1.0 - inter_weight / total_weight


def cluster_size_score(labels, min_frac=0.05, alpha=2.0):
    """
    Cluster size / cardinality quality score in [0,1].

    Args:
        labels : array-like of shape (n_samples,)
        min_frac : minimum acceptable cluster size as fraction of n
        alpha : imbalance penalty strength (>=1)

    Returns:
        float : size-based clustering score (1 = good sizes)
    """
    labels = np.asarray(labels)
    n = len(labels)

    unique, counts = np.unique(labels, return_counts=True)
    fracs = counts / n

    # Hard penalty for tiny clusters
    tiny_penalty = np.clip(fracs / min_frac, 0, 1)

    # Balance penalty (entropy-like)
    balance = np.exp(-alpha * np.var(fracs))

    # Final score
    score = np.mean(tiny_penalty) * balance
    return float(score)


def silhouette_score(X, labels, metric='euclidean'):
    """
    Compute a silhouette score (separation score between 0 and 1 suitable for nonlinear clusters).

    Note: 
        Even with perfect labels, silhouette is low where clusters are close in space, because silhouette measures metric separability, not topological correctness.
    
    Args:
        X: ndarray, shape (n_samples, n_features)
        labels: cluster labels
        metric: distance metric, e.g., 'euclidean', 'cosine'
    
    Returns:
        separation_score: float in [0,1]
    """
    if len(np.unique(labels)) < 2:
        return 0.0  # silhouette undefined for 1 cluster
    
    score = silhouette_score(X, labels, metric=metric)  # ranges [-1, 1]
    # normalize to 0-1
    separation_score = (score + 1) / 2
    return separation_score


def find_very_small_max_ratio(nums, ratio=0.05):
    if not nums:
        return []

    max_val = max(nums)
    threshold = ratio * max_val

    return [i for i, x in enumerate(nums) if x < threshold]


def find_very_small_statistical(nums, k=2):
    import statistics
    if len(nums) < 2:
        return []

    mean = statistics.mean(nums)
    stdev = statistics.stdev(nums)

    threshold = mean - k * stdev
    return [i for i, x in enumerate(nums) if x < threshold]