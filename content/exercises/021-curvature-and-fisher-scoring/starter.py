"""Curvature and Fisher scoring.

The loss bends by `X.T @ diag(A''(eta)) @ X`, where each row's variance is its
weight. Newton's step with that table fits every family with one loop. You
write three functions, in this order:

1. `glm_fisher_information`: the bend table.
2. `newton_step`: the step that solves `info @ delta = grad`.
3. `fit_fisher_scoring`: the loop.

The functions from earlier chapters are already here, solved.
"""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


# --- Written in the chapter "One dial, three distributions". Provided, already solved. ---


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return 0.5 * eta ** 2
    if family == "bernoulli":
        return np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def mean_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return eta.copy()
    if family == "bernoulli":
        return _sigmoid(eta)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def variance_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return np.ones_like(eta)
    if family == "bernoulli":
        p = _sigmoid(eta)
        return p * (1.0 - p)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


# --- Written in the chapter "One loss and one gradient for every family". Provided, already solved. ---


def glm_loss(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ w
    return float(np.sum(log_partition(family, eta) - y * eta))


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = mean_from_natural(family, X @ w)
    return X.T @ (mu - y)


# --- Write these in this chapter. ---


def glm_fisher_information(family: str, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute the bend table X.T @ diag(variance) @ X of the loss at `w`.

    The variance of each row is `variance_from_natural(family, X @ w)`. Scale
    the rows of `X` by it instead of building an n-by-n diagonal matrix.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Weights, shape (d,).
        X: Features, shape (n, d).

    Returns:
        A symmetric array of shape (d, d).
    """
    raise NotImplementedError("glm_fisher_information")


def newton_step(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the Fisher scoring step delta that solves `info @ delta = grad`.

    The caller updates the weights with `w = w - delta`. Use `np.linalg.solve`
    rather than an explicit inverse.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Weights, shape (d,).
        X: Features, shape (n, d).
        y: Labels, shape (n,).

    Returns:
        The step, shape (d,). It is the step, not the new weights.
    """
    raise NotImplementedError("newton_step")


def fit_fisher_scoring(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Fit a model by Fisher scoring, starting from zero weights.

    In each pass, append `glm_loss` to the history, stop without stepping if
    the gradient norm is below `tol`, and otherwise set `w = w - delta`.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        X: Features, shape (n, d).
        y: Labels, shape (n,).
        max_iter: Maximum number of steps.
        tol: Stop when the gradient norm is below this value.

    Returns:
        A tuple `(w, history)`: the weights, shape (d,), and the loss recorded
        before each step, so `history[0]` is the loss at zero weights.
    """
    raise NotImplementedError("fit_fisher_scoring")


if __name__ == "__main__":
    # The three hourly counts at hours 0, 1, 2: the running example.
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, 2.0])
    w0 = np.zeros(2)
    print("information at zero weights:\n", glm_fisher_information("poisson", w0, X), "\n(expected [[3, 3], [3, 5]])")
    print("first step delta:", newton_step("poisson", w0, X, y), "(expected [1/6, -0.5])")
    w, history = fit_fisher_scoring("poisson", X, y, max_iter=30)
    print("weights:", np.round(w, 5))
    print("losses: ", [round(v, 4) for v in history])
