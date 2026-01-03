import numpy as np

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