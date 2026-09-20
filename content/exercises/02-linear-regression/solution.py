"""Reference solution for the least squares exercise."""

import numpy as np
from scipy.linalg import solve_triangular


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    gram = X.T @ X
    rhs = X.T @ y
    lower = np.linalg.cholesky(gram)
    z = solve_triangular(lower, rhs, lower=True)
    return solve_triangular(lower.T, z, lower=False)


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(X, mode="reduced")
    return solve_triangular(r, q.T @ y, lower=False)


def fit_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros(X.shape[1])
    keep = s > rcond * s[0]
    # Dropping the small singular values puts zero weight on the null space,
    # which is what makes this the minimum-norm element of the solution set.
    return vt[keep].T @ ((u[:, keep].T @ y) / s[keep])


def residual_orthogonality(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(np.abs(X.T @ (y - X @ w)).max())


def condition_number(X: np.ndarray) -> float:
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0 or s[-1] == 0.0:
        return float("inf")
    return float(s[0] / s[-1])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0, -0.25])
    y = X @ w_true + 0.1 * rng.normal(size=n)

    print(f"condition number: {condition_number(X):.3f}")
    for name, w in [
        ("normal equations", fit_normal_equations(X, y)),
        ("qr", fit_qr(X, y)),
        ("svd", fit_svd(X, y)),
    ]:
        error = np.abs(w - w_true).max()
        print(
            f"{name:>17}: max coefficient error {error:.2e}, "
            f"orthogonality {residual_orthogonality(X, y, w):.2e}"
        )
