import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from utils.wfr import WFR

# Generate two moons dataset
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# Test: another circle, slightly offset
X_test, y_test_true = make_moons(n_samples=50, noise=0.08, random_state=24)
# X_test += np.array([0.01, 0.01])  # shift test data

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

# Predict test data using trained WFR
labels_test = wfr.predict(X_test)
adjacency_test = wfr.adjacency_test_

# Define colors for clusters:
cluster_colors = np.array(['orange', 'blue', 'black'])
labels = np.where(labels==-1, 2, labels)
labels_test = np.where(labels_test == -1, 2, labels_test)

# plot figure:
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# draw adjacency lines:
draw_adjacency_lines = False
if draw_adjacency_lines:
    n = X.shape[0]
    for i in range(n):
        for j in range(i+1, n):  # only upper triangle to avoid double drawing
            if adjacency[i, j]:
                ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color='gray', alpha=0.8, linewidth=0.7)

# draw points:
ax.scatter(X[:,0], X[:,1], c=cluster_colors[labels], s=50, alpha=0.3)

# Plot test points with different marker
ax.scatter(X_test[:, 0], X_test[:, 1], c=cluster_colors[labels_test], edgecolors="black", s=150, marker='X', label='Test')

# draw adjacency-test lines:
draw_adjacency_test_lines = True
if draw_adjacency_test_lines:
    n = X.shape[0]
    n_test = X_test.shape[0]
    for i in range(n_test):
        for j in range(n):
            if adjacency_test[i, j]:
                ax.plot([X_test[i, 0], X[j, 0]], [X_test[i, 1], X[j, 1]], color='black', alpha=0.8, linewidth=2)

# axes.set_title("WFR")
plt.xticks(())
plt.yticks(())

plt.show()
