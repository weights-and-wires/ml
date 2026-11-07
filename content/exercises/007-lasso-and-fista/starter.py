"""The lasso: shrink one number and zero it out, then fit any design with ISTA and FISTA.

Fill in the six functions below, in order. The first three handle a single
number, score a set of weights, and solve the lasso when X is the identity. The
last three fit the lasso for any design: ISTA, the KKT test, then FISTA.
`lipschitz_constant` is already written.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Shrink every entry towards 0 by `thresh`, and set it to 0 if it is closer than that.

    Args:
        x: Input array, any shape.
        thresh: Non-negative threshold.

    Returns:
        sign(x) * max(|x| - thresh, 0), entry by entry, with the same shape as x.
    """
    raise NotImplementedError("soft_threshold")


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Score the weights: half the squared misses plus lambda times the absolute size.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        lam: Lasso fee, lam >= 0.

    Returns:
        0.5 * ||X w - y||^2 + lam * sum(|w|) as a Python float. The fee has no half.
    """
    raise NotImplementedError("lasso_objective")


def lasso_identity(y: np.ndarray, lam: float) -> np.ndarray:
    """Return the exact lasso solution when the design matrix is the identity.

    Args:
        y: Targets, shape (d,). With X = I the predictions are the weights.
        lam: Lasso fee, lam >= 0.

    Returns:
        The weights that minimise 0.5 * ||w - y||^2 + lam * sum(|w|), shape (d,).
    """
    raise NotImplementedError("lasso_identity")


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
