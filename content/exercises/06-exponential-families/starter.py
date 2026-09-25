"""Exponential families and Fisher scoring for GLMs.

Three families share one set of formulas: `"gaussian"` with variance 1,
`"bernoulli"`, and `"poisson"`. Each one is described by a single function,
the log-partition function A(eta):

    p(y | eta) = h(y) * exp(eta * y - A(eta))

You write six functions, in this order:

1. `log_partition`: A(eta) for each family.
2. `mean_from_natural`: the mean, which is the first derivative A'(eta).
3. `variance_from_natural`: the variance, the second derivative A''(eta).
4. `glm_gradient`: X.T @ (mu - y), the same for every family.
5. `glm_fisher_information`: X.T @ diag(variance) @ X.
6. `fit_fisher_scoring`: the Newton loop from lesson 5, using the two above.

Shapes: X is (n, d), one row per example; y is (n,); w is (d,). The link is
canonical, so the natural parameter is the score itself: eta = X @ w.
"""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    """Compute the log-partition function A(eta) for one exponential family.

    Args:
        family: One of `"gaussian"` (unit dispersion), `"bernoulli"`, or
            `"poisson"`.
        eta: Natural parameter values, any shape.

    Returns:
        A(eta), same shape as `eta`. Stable for eta as large in magnitude as
        the mean and variance checks in this exercise require.
    """
    raise NotImplementedError("log_partition")


def mean_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    """Compute the mean E[y] = A'(eta) implied by the natural parameter.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Natural parameter values, any shape.

    Returns:
        The mean, same shape as `eta`.
    """
    raise NotImplementedError("mean_from_natural")


def variance_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    """Compute the variance Var[y] = A''(eta) implied by the natural parameter.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Natural parameter values, any shape.

    Returns:
        The variance, same shape as `eta`.
    """
    raise NotImplementedError("variance_from_natural")


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the gradient of the GLM negative log-likelihood, canonical link.

    With the canonical link, the natural parameter is `eta = X @ w`, and the
    gradient of the negative log-likelihood is `X.T @ (mu - y)` for every
    family in this exponential family, where `mu` is the mean implied by
    `eta`.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Coefficient vector, shape (d,).
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,), matching the support of `family`.

    Returns:
        The gradient, shape (d,).
    """
    raise NotImplementedError("glm_gradient")


def glm_fisher_information(family: str, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute the Fisher information of the GLM at `w`, canonical link.

    With the canonical link this equals `X.T @ diag(V(eta)) @ X`, where
    `V = variance_from_natural(family, eta)` and `eta = X @ w`.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Coefficient vector, shape (d,).
        X: Design matrix, shape (n, d).

    Returns:
        The Fisher information matrix, shape (d, d).
    """
    raise NotImplementedError("glm_fisher_information")


def fit_fisher_scoring(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Fit a canonical-link GLM by Fisher scoring, starting from w = 0.

    Each step solves `I(w) delta = g(w)` for the Fisher information `I` and
    the gradient `g`, and updates `w -= delta`. Record, before each step, the
    value `sum(log_partition(family, eta) - y * eta)`, which is the negative
    log-likelihood up to an additive constant that does not depend on `w`.
    Stop early, without taking that step, once the gradient norm falls below
    `tol`.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,), matching the support of `family`.
        max_iter: Maximum number of steps.
        tol: Stop when the gradient norm drops below this value.

    Returns:
        A tuple `(w, history)`: the fitted coefficients, shape (d,), and the
        list of loss values recorded before each step.

    Steps:
        1. Start from w = zeros(d) and an empty history list.
        2. In each pass, compute eta, append the loss, and compute the
           gradient. Stop if its norm is below `tol`.
        3. Otherwise solve the Fisher information system for the step and
           subtract it from w.
    """
    raise NotImplementedError("fit_fisher_scoring")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 400, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))])
    w_true = np.array([0.3, 0.5, -0.4])
    y = rng.poisson(np.exp(X @ w_true)).astype(float)

    w_hat, history = fit_fisher_scoring("poisson", X, y, max_iter=30)
    print(f"fisher scoring: {len(history)} iterations")
    print(f"true coefficients:      {w_true}")
    print(f"recovered coefficients: {w_hat}")
    print(f"max absolute error:     {np.abs(w_hat - w_true).max():.3f}")
