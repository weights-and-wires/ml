"""Reference solution: find the bottom of the loss bowl, and check it with right angles."""

import numpy as np


def add_ones_column(x: np.ndarray) -> np.ndarray:
    """Build the table X from sizes. This one is written for you.

    Args:
        x: Inputs, shape (n,).

    Returns:
        A table of shape (n, 2) whose first column is all ones and whose
        second column is x.
    """
    return np.column_stack([np.ones(len(x)), x])


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Return half the sum of squared misses, 0.5 * sum((X w - y) ** 2).

    Args:
        X: Table of inputs, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).

    Returns:
        One float.
    """
    misses = X @ w - y
    return float(0.5 * np.sum(misses**2))


def gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the slope of the loss along every weight, X^T (X w - y).

    Args:
        X: Table of inputs, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).

    Returns:
        A vector of shape (d,). It points uphill, so it is zero at the bottom.
    """
    return X.T @ (X @ w - y)


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the weights that set the gradient to zero.

    Solve X^T X w = X^T y with a linear solve. Do not form an inverse.

    Args:
        X: Table of inputs, shape (n, d), with full column rank.
        y: Targets, shape (n,).

    Returns:
        The weights, shape (d,).
    """
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
    r = y - X @ w
    return float(np.abs(X.T @ r).max())


def project(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the fitted values: the projection of y onto the columns of X.

    Args:
        X: Table of inputs, shape (n, d), with full column rank.
        y: Targets, shape (n,).

    Returns:
        The fitted values X w for the best weights w, shape (n,).
    """
    return X @ fit_normal_equations(X, y)


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
    scale = float(np.abs(X.T @ y).max())
    return residual_orthogonality(X, y, w) <= tol * scale


if __name__ == "__main__":
    sizes = np.array([1.0, 2.0, 3.0])
    prices = np.array([1.0, 2.0, 2.0])
    X = add_ones_column(sizes)

    trial = np.array([0.0, 1.0])
    print("loss at (0, 1):", loss(X, prices, trial))  # 0.5
    print("gradient at (0, 1):", gradient(X, prices, trial))  # [1. 3.]

    w = fit_normal_equations(X, prices)
    print("best weights:", w)  # about [0.6667 0.5]
    print("loss at the best weights:", loss(X, prices, w))  # about 0.0833
    print("gradient there:", gradient(X, prices, w))  # about [0. 0.]

    for candidate in (np.array([0.0, 1.0]), np.array([1.0, 0.0]), w):
        print(
            candidate,
            "check:", residual_orthogonality(X, prices, candidate),
            "best fit:", is_best_fit(X, prices, candidate),
        )  # 3.0, 5.0, about 0.0
    print("fitted values:", project(X, prices))  # about [1.1667 1.6667 2.1667]
