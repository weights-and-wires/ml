"""Reference solution for the ridge SVD path and degrees of freedom."""

from __future__ import annotations

import numpy as np


# Written in the chapter "Ridge: a fee on large weights".
def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide, returning 0 where the denominator is exactly 0."""
    return np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def shrink_factors(s: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    den = s[None, :] ** 2 + lambdas[:, None]
    return _safe_divide(np.broadcast_to(s[None, :], den.shape), den)


def ridge_svd_path(
    U: np.ndarray, s: np.ndarray, Vt: np.ndarray, y: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    c = U.T @ y  # how much of y lies along each direction, shape (r,)
    factors = shrink_factors(s, lambdas)  # shape (m, r)
    return (factors * c[None, :]) @ Vt


def effective_degrees_of_freedom(s: np.ndarray, lam) -> np.ndarray | float:
    s = np.asarray(s, dtype=float)
    lam_arr = np.asarray(lam, dtype=float)
    if lam_arr.ndim == 0:
        return float(np.sum(_safe_divide(s**2, s**2 + lam_arr)))
    den = s[None, :] ** 2 + lam_arr[:, None]
    return _safe_divide(np.broadcast_to(s[None, :] ** 2, den.shape), den).sum(axis=-1)


if __name__ == "__main__":
    # The two-direction example from the chapter: a strong direction and a weak one.
    X = np.array([[3.0, 3.0], [-0.1, 0.1]])
    y = np.array([6.0, 0.4])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    path = ridge_svd_path(U, s, Vt, y, lambdas)
    dof = effective_degrees_of_freedom(s, lambdas)
    print("singular values:", np.round(s, 4))
    print("lambda    w1       w2     degrees of freedom")
    for lam, w, df in zip(lambdas, path, dof):
        print(f"{lam:6.2f} {w[0]:8.3f} {w[1]:8.3f} {df:10.3f}")
    print("same as one solve per lambda:", np.allclose(path[3], ridge_closed_form(X, y, 1.0)))
