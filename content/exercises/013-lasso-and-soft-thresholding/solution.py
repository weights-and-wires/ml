"""Reference solution for soft thresholding and the lasso on the identity."""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    return float(0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w)))


def lasso_identity(y: np.ndarray, lam: float) -> np.ndarray:
    return soft_threshold(y, lam)


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
