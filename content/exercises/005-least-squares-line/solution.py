"""Reference solution for the least squares line exercise."""

import numpy as np


def add_ones_column(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    misses = X @ w - y
    return float(0.5 * np.sum(misses**2))


def gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X.T @ (X @ w - y)


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X, X.T @ y)


if __name__ == "__main__":
    sizes = np.array([1.0, 2.0, 3.0])
    prices = np.array([1.0, 2.0, 2.0])
    X = add_ones_column(sizes)

    trial = np.array([0.0, 1.0])
    print("loss at (0, 1):", loss(X, prices, trial))
    print("gradient at (0, 1):", gradient(X, prices, trial))

    w = fit_normal_equations(X, prices)
    print("best weights:", w)
    print("loss at the best weights:", loss(X, prices, w))
    print("gradient there:", gradient(X, prices, w))
