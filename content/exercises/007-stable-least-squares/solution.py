"""Reference solution for the stable least squares exercise."""

import numpy as np
from scipy.linalg import solve_triangular


def make_ill_conditioned(n: int, d: int, kappa: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.logspace(0.0, -np.log10(kappa), d)
    return u @ np.diag(s) @ v.T


def condition_number(X: np.ndarray) -> float:
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0 or s[-1] == 0.0:
        return float("inf")
    return float(s[0] / s[-1])


def digits_kept(kappa: float, squared: bool) -> float:
    k = float(np.log10(kappa))
    return max(0.0, 16.0 - (2.0 * k if squared else k))


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    lower = np.linalg.cholesky(X.T @ X)
    z = solve_triangular(lower, X.T @ y, lower=True)
    return solve_triangular(lower.T, z, lower=False)


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
    print("condition number of the houses:", condition_number(houses))
    print("normal equations:", fit_normal_equations(houses, y))
    print("QR:", fit_qr(houses, y))
    print("SVD:", fit_svd(houses, y))

    near = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]])
    for t in (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.03])):
        print("weights:", fit_normal_equations(near, t))

    repeated = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    print("rank of the repeated columns:", rank(repeated))
    print("shortest weights:", fit_svd(repeated, np.ones(3)))
