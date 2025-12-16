import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from utils.wfr import WFR

# Generate two moons dataset
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# WFR clustering:
resemblance_threshold = None
mark_outliers_as_minus_one = True
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

# Define colors for clusters:
cluster_colors = np.array(['orange', 'blue', 'black'])
labels = np.where(labels==-1, 2, labels)

# plot figure:
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# draw adjacency lines:
n = X.shape[0]
for i in range(n):
    for j in range(i+1, n):  # only upper triangle to avoid double drawing
        if adjacency[i, j]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='gray', alpha=0.8, linewidth=0.7)

# draw points:
ax.scatter(X[:,0], X[:,1], c=cluster_colors[labels], s=50)

# axes.set_title("WFR")
plt.xticks(())
plt.yticks(())

plt.show()
