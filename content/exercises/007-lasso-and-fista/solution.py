"""Reference solution for the lasso: soft thresholding, ISTA, FISTA, and the KKT test."""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    return float(0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w)))


def lasso_identity(y: np.ndarray, lam: float) -> np.ndarray:
    return soft_threshold(y, lam)


# Provided: the largest curvature of the squared error, so 1 / L is a safe step.
def lipschitz_constant(X: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(X.T @ X).max())


def ista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    history = np.empty(iters + 1)
    history[0] = lasso_objective(X, y, w, lam)
    for k in range(iters):
        grad = X.T @ (X @ w - y)
        w = soft_threshold(w - step * grad, step * lam)
        history[k + 1] = lasso_objective(X, y, w, lam)
    return w, history


def kkt_violation(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    g = X.T @ (X @ w - y)
    nonzero = w != 0
    breach = np.where(nonzero, np.abs(g + lam * np.sign(w)), np.maximum(np.abs(g) - lam, 0.0))
    return float(breach.max())


def fista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    z = w.copy()
    t = 1.0
    history = np.empty(iters + 1)
    history[0] = lasso_objective(X, y, w, lam)
    for k in range(iters):
        grad = X.T @ (X @ z - y)
        w_new = soft_threshold(z - step * grad, step * lam)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        history[k + 1] = lasso_objective(X, y, w, lam)
    return w, history


if __name__ == "__main__":
    # The identity design: ridge shrinks, the lasso shrinks and kills.
    y = np.array([3.0, 1.0])
    X = np.eye(2)
    ridge = np.linalg.solve(X.T @ X + 1.0 * np.eye(2), X.T @ y)
    lasso = lasso_identity(y, 1.0)
    print("ridge weights, lambda = 1:", ridge)
    print("lasso weights, lambda = 1:", lasso)
    print("lasso score at the lasso weights:", lasso_objective(X, y, lasso, 1.0))
    print("lasso score at the ridge weights:", lasso_objective(X, y, ridge, 1.0))
    print("soft_threshold at 1:", soft_threshold(np.array([3.0, -3.0, 0.5, -0.5, 0.0, 2.0]), 1.0))

    # The one-column example from the chapter.
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    w, history = ista(X, y, lam=1.0, step=1.0 / lipschitz_constant(X), iters=3)
    print("one column: weight", w, "scores", np.round(history, 3))
    print("kkt breach at the answer:", kkt_violation(X, y, w, 1.0), " at zero:", kkt_violation(X, y, np.zeros(1), 1.0))

    # Correlated columns: 40 features, 5 of which matter.
    rng = np.random.default_rng(0)
    n, d = 100, 40
    X = rng.normal(size=(n, 5)) @ rng.normal(size=(5, d)) + 0.5 * rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[:5] = rng.normal(size=5) * 2.0
    y = X @ w_true + 0.1 * rng.normal(size=n)
    step = 1.0 / lipschitz_constant(X)
    _, slow = ista(X, y, 3.0, step, 80)
    w, fast = fista(X, y, 3.0, step, 80)
    print(f"after 80 steps: ISTA score {slow[-1]:.3f}, FISTA score {fast[-1]:.3f}")
    print("weights set to exactly zero by FISTA:", int(np.sum(w == 0)), "of", d)
