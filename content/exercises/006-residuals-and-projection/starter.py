"""Check a fit with right angles.

You write three functions. The residual of a fit is r = y - X w. At the best
weights, r is orthogonal to every column of X, so X^T r = 0.
"""

import numpy as np


# You wrote this function in the chapter "Fit a line by minimising squared
# misses". It is included so that `project` can use it.
def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X, X.T @ y)


def residual_orthogonality(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Return the largest absolute entry of X^T (y - X w).

    Args:
        X: Table of inputs, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).

    Returns:
        One float. It is zero, up to rounding, at the best weights.
    """
    raise NotImplementedError("residual_orthogonality")


def project(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the fitted values: the projection of y onto the columns of X.

    Args:
        X: Table of inputs, shape (n, d), with full column rank.
        y: Targets, shape (n,).

    Returns:
        The fitted values X w for the best weights w, shape (n,).
    """
    raise NotImplementedError("project")


def is_best_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True when w passes the right-angle check at a relative tolerance.

    The check passes when residual_orthogonality(X, y, w) is at most
    tol * max(abs(X^T y)). This keeps the answer the same if you rescale y.

    Args:
        X: Table of inputs, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        tol: Relative tolerance.

    Returns:
        True when the weights pass, otherwise False.
    """
    raise NotImplementedError("is_best_fit")


if __name__ == "__main__":
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])

    for w in (np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([2 / 3, 1 / 2])):
        print(
            w,
            "check:", residual_orthogonality(X, y, w),
            "best fit:", is_best_fit(X, y, w),
        )  # 3.0, 5.0, about 0.0
    print("fitted values:", project(X, y))  # about [1.1667 1.6667 2.1667]
