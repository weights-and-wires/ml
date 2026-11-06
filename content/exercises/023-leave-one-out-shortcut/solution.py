"""Leave-one-out error from one ridge fit.

You write three functions, in this order:

1. `hat_matrix`: the n-by-n matrix H that turns the targets into ridge fitted values.
2. `leverages`: how much each point controls its own fitted value, the diagonal of H.
3. `loo_linear_smoother`: the leave-one-out mean squared error from one fit.

A fourth function, `loo_by_refitting`, is provided. It does the same job the slow
way, with n refits, so you can compare your shortcut against it.
Shapes: X is (n, d), one row per point; y is (n,).
"""

import numpy as np


def hat_matrix(X: np.ndarray, lam: float) -> np.ndarray:
    """Build H = X (X^T X + lam I)^{-1} X^T for ridge regression.

    H "puts a hat on y": `H @ y` is the vector of fitted values of the ridge
    fit on all n points.

    Args:
        X: Design matrix, shape (n, d).
        lam: Nonnegative ridge penalty.

    Returns:
        The hat matrix, shape (n, n).
    """
    d = X.shape[1]
    return X @ np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T


def leverages(X: np.ndarray, lam: float) -> np.ndarray:
    """Return how much each point controls its own fitted value.

    Args:
        X: Design matrix, shape (n, d).
        lam: Nonnegative ridge penalty.

    Returns:
        The diagonal of the hat matrix, shape (n,).
    """
    return np.diag(hat_matrix(X, lam))


def loo_linear_smoother(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Compute the leave-one-out mean squared error of ridge from one fit.

    With r = y - H y the in-sample residuals and h the leverages, the
    leave-one-out residual of point i is r_i / (1 - h_i).

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Nonnegative ridge penalty.

    Returns:
        The mean of the squared leave-one-out residuals, one float.
    """
    H = hat_matrix(X, lam)
    residual = y - H @ y
    return float(np.mean((residual / (1.0 - np.diag(H))) ** 2))


# Provided, not for you to write: the slow way, with n separate fits.
def loo_by_refitting(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Leave-one-out mean squared error by fitting ridge n times."""
    n, d = X.shape
    errors = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        w = np.linalg.solve(X[keep].T @ X[keep] + lam * np.eye(d), X[keep].T @ y[keep])
        errors[i] = (y[i] - X[i] @ w) ** 2
    return float(np.mean(errors))


if __name__ == "__main__":
    # The chapter's hand example: inputs 1, 2, 3, targets 1, 2, 2, penalty 1.
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 2.0])
    print("leverages:", np.round(leverages(X, 1.0), 4), "(want 1/15, 4/15, 9/15)")
    print("in-sample residuals:", np.round(y - hat_matrix(X, 1.0) @ y, 4))
    print(f"leave-one-out MSE, shortcut:  {loo_linear_smoother(X, y, 1.0):.4f}")
    print(f"leave-one-out MSE, refitting: {loo_by_refitting(X, y, 1.0):.4f}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = X @ rng.normal(size=6) + rng.normal(size=200)
    for lam in (0.1, 1.0, 10.0):
        print(f"lam={lam:<5} shortcut {loo_linear_smoother(X, y, lam):.6f}"
              f"  refitting {loo_by_refitting(X, y, lam):.6f}")
