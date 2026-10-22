"""Reference solution for the ridge penalty and its closed form."""

from __future__ import annotations

import numpy as np


def ridge_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    misses = X @ w - y
    return float(0.5 * np.sum(misses**2) + 0.5 * lam * np.dot(w, w))


def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    return X.T @ (X @ w - y) + lam * w


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


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
