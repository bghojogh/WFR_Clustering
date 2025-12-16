import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from utils.wfr import WFR
from tqdm import tqdm
import os

def make_two_spirals(n_points=1000, noise=0.5, factor=0.75, random_points=False):
    """
    Generates the two-spirals dataset.

    Args:
        n_points (int): Number of points per spiral. Total points will be 2 * n_points.
        noise (float): Amount of random noise to add to the coordinates.
        factor (float): Scaling factor for the spiral radius.

    Returns:
        tuple: (X, y) where X is the data matrix (coordinates)
               and y is the labels vector.
    """
    # Total points for one spiral
    if random_points:
        # note: random points may have gaps within each spiral, resulting in more than two clusters
        n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    else:
        offset = 2  # additive constant to offset the spiral from the origin.
        max_angle = 780 * (2 * np.pi) / 360
        n = np.linspace(offset, max_angle + offset, n_points).reshape(n_points, 1)

    # Spiral 1 coordinates
    # The 'factor' controls how tightly the spiral is wound.
    d1x = -np.cos(n) * n * factor + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n * factor + np.random.rand(n_points, 1) * noise

    # Spiral 2 coordinates (flipped version of Spiral 1)
    d2x = np.cos(n) * n * factor + np.random.rand(n_points, 1) * noise
    d2y = -np.sin(n) * n * factor + np.random.rand(n_points, 1) * noise

    # X is the combined data (coordinates)
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((d2x, d2y))))

    # y is the label (0 for spiral 1, 1 for spiral 2)
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))

    return X, y

# Generate two moons dataset
# X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
X, y_true = make_two_spirals(n_points=500*2, random_points=False)

# create a 10x10 grid
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
axes = axes.flatten()

resemblance_threshold_grid_search_step = 0.01
for idx, resemblance_threshold in enumerate(tqdm(np.arange(1.0, 0, -resemblance_threshold_grid_search_step), desc="resemblance_threshold")):
    ax = axes[idx]   # <-- use subplot

    # WFR clustering:
    mark_outliers_as_minus_one = False
    knn_k = 10
    resemblance_measure = "log_based"   # options: log_based, cosine, kernel
    kernel = "rbf"   # options: rbf, sigmoid, poly, cosine, linear --> note: rbf works best for Kernel WFR
    wfr = WFR(resemblance_threshold=resemblance_threshold, resemblance_measure=resemblance_measure, mark_outliers_as_minus_one=mark_outliers_as_minus_one, knn_k=knn_k, kernel=kernel)
    if resemblance_measure == "kernel":
        wfr_algorithm_name = "Kernel WFR"
    else:
        wfr_algorithm_name = "WFR"

    labels = wfr.fit_predict(X)
    adjacency = wfr.adjacency_

    # draw adjacency lines:
    draw_adjacency_lines = False
    if draw_adjacency_lines:
        n = X.shape[0]
        for i in range(n):
            for j in range(i+1, n):  # only upper triangle to avoid double drawing
                if adjacency[i, j]:
                    ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='gray', alpha=0.8, linewidth=0.7)

    # draw points:
    X_inlier = X[labels!=-1, :]
    X_outlier = X[labels==-1, :]
    labels_inlier = labels[labels!=-1]
    ax.scatter(X_inlier[:,0], X_inlier[:,1], c=labels_inlier, s=10, cmap='tab10')
    ax.scatter(X_outlier[:,0], X_outlier[:,1], c="black", s=10)

    # title per subplot
    ax.set_title(f"τ={resemblance_threshold:.2f}", fontsize=15)
    ax.set_xticks(())
    ax.set_yticks(())

# layout and save
plt.tight_layout()
os.makedirs("./img/toy_example4/", exist_ok=True)
plt.savefig("./img/toy_example4/wfr_threshold_grid.png", dpi=200)
plt.show()
