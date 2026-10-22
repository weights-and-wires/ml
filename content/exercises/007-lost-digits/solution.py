"""Reference solution for the lost digits exercise."""

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


if __name__ == "__main__":
    houses = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    print("condition number of the houses:", condition_number(houses))

    near = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]])
    print("condition number, nearly repeated:", condition_number(near))
    for y in (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.03])):
        print("weights:", fit_normal_equations(near, y))

    for kappa in (1e3, 1e7):
        print(
            f"kappa {kappa:.0e}: about {digits_kept(kappa, False):.0f} digits with X, "
            f"{digits_kept(kappa, True):.0f} through X^T X"
        )
