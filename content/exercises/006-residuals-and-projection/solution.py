"""Reference solution for the residuals and projection exercise."""

import numpy as np


# You wrote this function in the chapter "Fit a line by minimising squared
# misses". It is included so that `project` can use it.
def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X, X.T @ y)


def residual_orthogonality(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    r = y - X @ w
    return float(np.abs(X.T @ r).max())


def project(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X @ fit_normal_equations(X, y)


def is_best_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, tol: float = 1e-8) -> bool:
    scale = float(np.abs(X.T @ y).max())
    return residual_orthogonality(X, y, w) <= tol * scale


if __name__ == "__main__":
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])

    for w in (np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([2 / 3, 1 / 2])):
        print(
            w,
            "check:", residual_orthogonality(X, y, w),
            "best fit:", is_best_fit(X, y, w),
        )
    print("fitted values:", project(X, y))
