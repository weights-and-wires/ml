"""One loss and one gradient for every family.

A linear score drives the dial, `eta = X @ w`, and the prediction is the mean
`mu = A'(eta)`. The loss is `sum(A(eta) - y * eta)` and its gradient is
`X.T @ (mu - y)`. You write three functions, in this order:

1. `glm_loss`: the loss.
2. `glm_gradient`: the gradient.
3. `gradient_descent_glm`: plain gradient steps from zero weights.

The three functions from the previous chapter are already here, solved.
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


# --- Write these in this chapter. ---


def glm_loss(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Compute the loss sum(A(eta) - y * eta) with eta = X @ w.

    This is the negative log-likelihood with the part that does not depend on
    `w` dropped. It equals half the squared error plus a constant for the
    Gaussian, and the log loss for the Bernoulli.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Weights, shape (d,).
        X: Features, shape (n, d).
        y: Labels, shape (n,).

    Returns:
        The loss, as a float. It sums over rows and does not average.
    """
    raise NotImplementedError("glm_loss")


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the gradient of `glm_loss` with respect to `w`.

    The gradient is `X.T @ (mu - y)`, where `mu` is the mean implied by
    `eta = X @ w`: prediction minus truth, for every family.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        w: Weights, shape (d,).
        X: Features, shape (n, d).
        y: Labels, shape (n,).

    Returns:
        The gradient, shape (d,).
    """
    raise NotImplementedError("glm_gradient")


def gradient_descent_glm(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    steps: int = 500,
) -> tuple[np.ndarray, list[float]]:
    """Fit a model by plain gradient steps, starting from zero weights.

    Each step sets `w = w - lr * gradient`.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        X: Features, shape (n, d).
        y: Labels, shape (n,).
        lr: Step size.
        steps: Number of steps to take.

    Returns:
        A tuple `(w, history)`: the final weights, shape (d,), and the loss at
        zero weights followed by the loss after each step, so `history` has
        `steps + 1` entries.
    """
    raise NotImplementedError("gradient_descent_glm")


if __name__ == "__main__":
    # Three hourly counts at hours 0, 1, 2: the running example.
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, 2.0])
    w0 = np.zeros(2)
    print("loss at zero weights:    ", glm_loss("poisson", w0, X, y), "(expected 3)")
    print("gradient at zero weights:", glm_gradient("poisson", w0, X, y), "(expected [-1, -2])")
    w, history = gradient_descent_glm("poisson", X, y, lr=0.05, steps=500)
    print("weights after 500 steps: ", np.round(w, 4))
    print("loss after 500 steps:    ", round(history[-1], 5))
