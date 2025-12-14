import numpy as np
from typing import Callable, Optional, List


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