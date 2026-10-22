"""Lasso: shrink one number and zero it out, then solve the lasso when X is the identity.

Fill in the three functions below, in order. The first handles a single number,
the second scores a set of weights, and the third puts them together.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Shrink every entry towards 0 by `thresh`, and set it to 0 if it is closer than that.

    Args:
        x: Input array, any shape.
        thresh: Non-negative threshold.

    Returns:
        sign(x) * max(|x| - thresh, 0), entry by entry, with the same shape as x.
    """
    raise NotImplementedError("soft_threshold")


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Score the weights: half the squared misses plus lambda times the absolute size.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        lam: Lasso fee, lam >= 0.

    Returns:
        0.5 * ||X w - y||^2 + lam * sum(|w|) as a Python float. The fee has no half.
    """
    raise NotImplementedError("lasso_objective")


def lasso_identity(y: np.ndarray, lam: float) -> np.ndarray:
    """Return the exact lasso solution when the design matrix is the identity.

    Args:
        y: Targets, shape (d,). With X = I the predictions are the weights.
        lam: Lasso fee, lam >= 0.

    Returns:
        The weights that minimise 0.5 * ||w - y||^2 + lam * sum(|w|), shape (d,).
    """
    raise NotImplementedError("lasso_identity")


if __name__ == "__main__":
    y = np.array([3.0, 1.0])
    X = np.eye(2)
    ridge = np.linalg.solve(X.T @ X + 1.0 * np.eye(2), X.T @ y)
    lasso = lasso_identity(y, 1.0)
    print("ridge weights, lambda = 1:", ridge)
    print("lasso weights, lambda = 1:", lasso)
    print("lasso score at the lasso weights:", lasso_objective(X, y, lasso, 1.0))
    print("lasso score at the ridge weights:", lasso_objective(X, y, ridge, 1.0))
    print("soft_threshold at 1:", soft_threshold(np.array([3.0, -3.0, 0.5, -0.5, 0.0, 2.0]), 1.0))
