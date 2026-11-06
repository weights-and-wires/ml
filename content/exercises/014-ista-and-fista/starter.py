"""Fit the lasso with ISTA and FISTA, and test whether you have reached the bottom.

Fill in the three functions below, in order: ISTA, the KKT test, then FISTA.
The helpers above them are already written.
"""

from __future__ import annotations

import numpy as np


# Written in the chapter "Lasso and soft thresholding: why some weights become exactly zero".
def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


# Written in the chapter "Lasso and soft thresholding: why some weights become exactly zero".
def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    return float(0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w)))


# Provided: the largest curvature of the squared error, so 1 / L is a safe step.
def lipschitz_constant(X: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(X.T @ X).max())


def ista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise 0.5 ||Xw - y||^2 + lam ||w||_1 by stepping downhill, then thresholding.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Lasso fee, lam >= 0.
        step: Step size. Use 1 / lipschitz_constant(X) or smaller.
        iters: Number of steps.
        w0: Starting weights, shape (d,). Defaults to zeros. Do not change the caller's array.

    Returns:
        (w, history). w has shape (d,). history has shape (iters + 1,), and
        history[k] is the lasso score after k steps, so history[0] is the start.
    """
    raise NotImplementedError("ista")


def kkt_violation(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Return the largest breach of the two KKT rules, which is 0 at the lasso answer.

    With g = X^T (X w - y): where w[i] is not 0 the rule is g[i] + lam * sign(w[i]) = 0,
    and where w[i] is 0 the rule is |g[i]| <= lam.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights to test, shape (d,).
        lam: Lasso fee, lam >= 0.

    Returns:
        The largest breach as a float: |g[i] + lam * sign(w[i])| for a nonzero weight,
        max(|g[i]| - lam, 0) for a zero weight.
    """
    raise NotImplementedError("kkt_violation")


def fista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the same lasso score as `ista`, with momentum.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Lasso fee, lam >= 0.
        step: Step size, as in `ista`.
        iters: Number of steps.
        w0: Starting weights, shape (d,). Defaults to zeros.

    Returns:
        (w, history), with the same shapes and meaning as in `ista`. The history
        records the score at w, not at the look-ahead point.
    """
    raise NotImplementedError("fista")


if __name__ == "__main__":
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
