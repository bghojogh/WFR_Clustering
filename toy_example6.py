import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

def draw_ovals(ax, X_spiral, scale_major=1.2, scale_minor=0.4, offset=0.0):
    """
    For each neighboring two points in the spirals, it circles them by an oval where the two points are its two centers.
    """
    for i in range(len(X_spiral) - 1):
        p1 = X_spiral[i]
        p2 = X_spiral[i + 1]

        # Midpoint
        center = (p1 + p2) / 2

        # Distance and orientation
        dist = np.linalg.norm(p2 - p1)
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        # Ellipse dimensions
        major = scale_major * dist + offset
        minor = scale_minor * dist + offset

        # Ellipse
        ellipse = Ellipse(
            xy=center,
            width=major,
            height=minor,
            angle=angle,
            edgecolor="black",
            facecolor="none",
            linewidth=1,
            alpha=0.6,
        )
        ax.add_patch(ellipse)

# settings:
n_points = 50  # per spiral
offset_of_points_from_oval_centers = 0.5

X, y_true = make_two_spirals(n_points=n_points, random_points=False, noise=0)
# plt.plot(X[:, 0], X[:, 1], "*")
# plt.show()

fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], "o")

# Spiral 1
draw_ovals(ax, X[:n_points], offset=offset_of_points_from_oval_centers)

# Spiral 2
draw_ovals(ax, X[n_points:], offset=offset_of_points_from_oval_centers)

ax.set_aspect("equal")
plt.xticks(())
plt.yticks(())
plt.show()