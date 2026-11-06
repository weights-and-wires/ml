"""Ridge regression: score a fit with a fee on large weights, then find the bottom.

Fill in the three functions below, in order. The first scores a set of weights,
the second gives the slope of that score, and the third solves for the weights
where the slope is zero.
"""

from __future__ import annotations

import numpy as np


def ridge_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Score the weights: half the squared misses plus half lambda times the squared length.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        lam: Ridge fee, lam >= 0.

    Returns:
        0.5 * ||X w - y||^2 + 0.5 * lam * ||w||^2 as a Python float.
    """
    raise NotImplementedError("ridge_objective")


def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Return the gradient of the ridge score with respect to the weights.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        lam: Ridge fee, lam >= 0.

    Returns:
        The gradient, shape (d,).
    """
    raise NotImplementedError("ridge_gradient")


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve for the weights at the bottom of the ridge bowl with one linear system.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Ridge fee, lam >= 0.

    Returns:
        The weights where the gradient is zero, shape (d,).
    """
    raise NotImplementedError("ridge_closed_form")


if __name__ == "__main__":
    # The two-point example from the chapter: one weight, labels 0 and 1.
    X = np.array([[0.0], [0.1]])
    y = np.array([0.0, 1.0])
    print("lambda   slope   score at the slope")
    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        w = ridge_closed_form(X, y, lam)
        print(f"{lam:6.2f}  {w[0]:6.3f}  {ridge_objective(X, y, w, lam):8.4f}")

    # Two identical columns: least squares cannot choose, ridge splits evenly.
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 2.0])
    print("identical columns, lambda = 1:", ridge_closed_form(X, y, 1.0))
