"""Fit a Poisson, Gaussian, or Bernoulli model with one loop.

Every function takes a family name, one of `"gaussian"`, `"bernoulli"`, or
`"poisson"`, and raises `ValueError` for any other name. You write eight
functions, in this order:

1. `log_partition`: A(eta), the log of the total that normalises the family.
2. `mean_from_natural`: A'(eta), the slope of A, which is the mean.
3. `variance_from_natural`: A''(eta), the bend of A, which is the variance.
4. `glm_loss`: the loss sum(A(eta) - y * eta) with eta = X @ w.
5. `glm_gradient`: its gradient, X.T @ (mu - y).
6. `glm_fisher_information`: the bend table X.T @ diag(variance) @ X.
7. `newton_step`: the step that solves `info @ delta = grad`.
8. `fit_fisher_scoring`: the loop that repeats w = w - delta from zero weights.

Shapes used everywhere: X is (n, d), one row per example; y is (n,); w is (d,).
"""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    """Return A(eta) for the family, elementwise.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Dial settings, an array of any shape.

    Returns:
        An array of the same shape: eta**2 / 2 for the Gaussian, softplus(eta)
        for the Bernoulli, and exp(eta) for the Poisson. The Bernoulli value
        must stay finite at eta = -800 and 800.
    """
    raise NotImplementedError("write log_partition")


def mean_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    """Return the mean A'(eta) for the family, elementwise.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Dial settings, an array of any shape.

    Returns:
        An array of the same shape: eta for the Gaussian, a sigmoid for the
        Bernoulli, and exp(eta) for the Poisson. The sigmoid must not overflow
        at eta = -800 or 800; `np.exp(-np.logaddexp(0, -eta))` is safe.
    """
    raise NotImplementedError("write mean_from_natural")


def variance_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    """Return the variance A''(eta) for the family, elementwise.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Dial settings, an array of any shape.

    Returns:
        An array of the same shape: ones for the Gaussian, p * (1 - p) for the
        Bernoulli, and exp(eta) for the Poisson. Return an array, not the bare
        number 1, for the Gaussian.
    """
    raise NotImplementedError("write variance_from_natural")


def glm_loss(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Return the loss sum(A(eta) - y * eta) with eta = X @ w, as one float.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Weights, shape (d,).
        X: Features, shape (n, d).
        y: Labels, shape (n,).

    Returns:
        A sum over rows, not a mean. On the three-row shop data at zero weights
        with the Poisson family it is 3.
    """
    raise NotImplementedError("write glm_loss")


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the gradient of `glm_loss` with respect to the weights.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Weights, shape (d,).
        X: Features, shape (n, d).
        y: Labels, shape (n,).

    Returns:
        X.T @ (mu - y), shape (d,), where mu is the mean of each row's dial.
        On the shop data at zero weights it is [-1, -2].
    """
    raise NotImplementedError("write glm_gradient")


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
    raise NotImplementedError("write glm_fisher_information")


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
    raise NotImplementedError("write newton_step")


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
    raise NotImplementedError("write fit_fisher_scoring")


if __name__ == "__main__":
    eta = np.array([-2.0, 0.0, 2.0])
    for family in FAMILIES:
        print(family)
        print("  A(eta)   :", np.round(log_partition(family, eta), 4))
        print("  mean     :", np.round(mean_from_natural(family, eta), 4))
        print("  variance :", np.round(variance_from_natural(family, eta), 4))

    # The three hourly counts at hours 0, 1, 2: the running example.
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, 2.0])
    w0 = np.zeros(2)
    print("loss at zero weights    :", glm_loss("poisson", w0, X, y), "(expected 3)")
    print("gradient at zero weights:", glm_gradient("poisson", w0, X, y), "(expected [-1, -2])")
    print("information at zero weights:\n", glm_fisher_information("poisson", w0, X), "\n(expected [[3, 3], [3, 5]])")
    print("first step delta:", newton_step("poisson", w0, X, y), "(expected [1/6, -0.5])")
    w, history = fit_fisher_scoring("poisson", X, y, max_iter=30)
    print("weights:", np.round(w, 5))
    print("losses: ", [round(v, 4) for v in history])
