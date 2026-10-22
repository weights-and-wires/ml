"""Reference solution for the QR and SVD exercise."""

import numpy as np
from scipy.linalg import solve_triangular


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(X, mode="reduced")
    return solve_triangular(r, q.T @ y, lower=False)


def fit_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros(X.shape[1])
    keep = s > rcond * s[0]
    # Dropping the tiny singular values puts zero weight on the null space,
    # which makes this the shortest of the equally good weight vectors.
    return vt[keep].T @ ((u[:, keep].T @ y) / s[keep])


def rank(X: np.ndarray, rcond: float = 1e-12) -> int:
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0:
        return 0
    return int(np.sum(s > rcond * s[0]))


if __name__ == "__main__":
    houses = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])
    print("QR:", fit_qr(houses, y))
    print("SVD:", fit_svd(houses, y))

    repeated = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    ones = np.ones(3)
    print("rank of the repeated columns:", rank(repeated))
    print("shortest weights:", fit_svd(repeated, ones))
