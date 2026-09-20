"""Reference solution for ridge, lasso, and the proximal view."""

from __future__ import annotations

import numpy as np


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def ridge_svd_path(
    U: np.ndarray, s: np.ndarray, Vt: np.ndarray, y: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    lambdas = np.asarray(lambdas, dtype=float)
    u_dot_y = U.T @ y  # shape (r,), computed once
    # shrinkage[j, i] = s_i / (s_i^2 + lambdas[j])
    shrinkage = s[None, :] / (s[None, :] ** 2 + lambdas[:, None])
    return (shrinkage * u_dot_y[None, :]) @ Vt


def effective_degrees_of_freedom(s: np.ndarray, lam) -> np.ndarray | float:
    lam_arr = np.asarray(lam, dtype=float)
    if lam_arr.ndim == 0:
        return float(np.sum(s ** 2 / (s ** 2 + lam_arr)))
    # Broadcast s over the leading axis of lam_arr: shape (len(lam), len(s)).
    dof = (s[None, :] ** 2 / (s[None, :] ** 2 + lam_arr[:, None])).sum(axis=-1)
    return dof


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def _lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    return float(0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w)))


def ista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    w = np.zeros(d) if w0 is None else w0.copy()
    history = np.empty(iters + 1)
    history[0] = _lasso_objective(X, y, w, lam)
    for k in range(iters):
        grad = X.T @ (X @ w - y)
        w = soft_threshold(w - step * grad, step * lam)
        history[k + 1] = _lasso_objective(X, y, w, lam)
    return w, history


def fista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    w = np.zeros(d) if w0 is None else w0.copy()
    z = w.copy()
    t = 1.0
    history = np.empty(iters + 1)
    history[0] = _lasso_objective(X, y, w, lam)
    for k in range(iters):
        grad = X.T @ (X @ z - y)
        w_new = soft_threshold(z - step * grad, step * lam)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w = w_new
        t = t_new
        history[k + 1] = _lasso_objective(X, y, w, lam)
    return w, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 100, 40
    # Correlated columns push the condition number up, so FISTA's advantage
    # over ISTA is visible after a modest number of iterations.
    latent = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = latent @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[:5] = rng.normal(size=5) * 2.0
    y = X @ w_true + 0.1 * rng.normal(size=n)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.array([0.1, 1.0, 10.0, 100.0])
    path = ridge_svd_path(U, s, Vt, y, lambdas)
    dof = effective_degrees_of_freedom(s, lambdas)
    print("ridge path, effective degrees of freedom:")
    for lam, w, df in zip(lambdas, path, dof):
        print(f"  lambda={lam:>7.2f}  dof={df:5.2f}  ||w||={np.linalg.norm(w):.3f}")

    L = np.linalg.eigvalsh(X.T @ X).max()
    lam_lasso = 3.0
    _, ista_history = ista(X, y, lam_lasso, 1.0 / L, 150)
    _, fista_history = fista(X, y, lam_lasso, 1.0 / L, 150)
    print(f"\nafter 150 steps: ISTA objective {ista_history[-1]:.4f}, FISTA {fista_history[-1]:.4f}")
