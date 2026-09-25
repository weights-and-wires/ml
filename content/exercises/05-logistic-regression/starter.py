"""Fit a logistic regression two ways.

You write six functions, in this order:

1. `sigmoid`: turn scores into probabilities without overflow.
2. `log_loss`: the loss the model minimises.
3. `gradient`: first derivatives of the loss, shape (d,).
4. `hessian`: second derivatives of the loss, shape (d, d).
5. `fit_gradient_descent`: fit by small fixed steps.
6. `fit_newton`: fit by Newton steps.

Shapes used everywhere: X is (n, d), one row per example; y is (n,) with
labels 0 or 1; w is (d,). The loss is a sum over examples, not a mean:

    L(w) = sum_i [softplus(z_i) - y_i * z_i] + (ridge / 2) * ||w||^2,
    where z = X @ w and softplus(z) = log(1 + exp(z)).

Compute softplus with the stable rewrite from the lesson, or the checks at
scores of +-800 fail.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid elementwise, stable for any finite input.

    Args:
        z: Array of any shape.

    Returns:
        An array of the same shape holding 1 / (1 + exp(-z)), with no overflow
        or invalid-value warning even when `z` holds entries as large in
        magnitude as 800.

    Steps:
        1. Convert `z` to a float array and make an empty output array.
        2. Build a mask for the entries where z >= 0.
        3. Fill each part with the formula that only exponentiates a number
           that is zero or negative.
    """
    raise NotImplementedError("sigmoid")


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    """Compute the ridge-penalised negative log-likelihood.

    Args:
        X: Design matrix, shape (n, d).
        y: Binary labels in {0, 1}, shape (n,).
        w: Coefficient vector, shape (d,).
        ridge: Nonnegative ridge coefficient, added as (ridge / 2) ||w||^2.

    Returns:
        The scalar loss, computed through log-sum-exp so it stays finite for
        any `w` and `X` whose product `X @ w` has entries as large in
        magnitude as 800.

    Steps:
        1. Compute the scores z = X @ w.
        2. Compute softplus(z) with the stable rewrite.
        3. Sum softplus(z) - y * z over all examples.
        4. Add the ridge penalty.
    """
    raise NotImplementedError("log_loss")


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Compute the gradient of `log_loss` with respect to `w`.

    Args:
        X: Design matrix, shape (n, d).
        y: Binary labels in {0, 1}, shape (n,).
        w: Coefficient vector, shape (d,).
        ridge: Nonnegative ridge coefficient.

    Returns:
        The gradient, shape (d,): X^T (sigmoid(X w) - y) + ridge * w.

    Steps:
        1. Compute the probabilities p from the scores.
        2. Multiply the errors by X transposed, then add the ridge term.
    """
    raise NotImplementedError("gradient")


def hessian(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Compute the Hessian of `log_loss` with respect to `w`.

    Args:
        X: Design matrix, shape (n, d).
        y: Binary labels in {0, 1}, shape (n,).
        w: Coefficient vector, shape (d,).
        ridge: Nonnegative ridge coefficient.

    Returns:
        The Hessian, shape (d, d): X^T diag(p (1 - p)) X + ridge * I, where
        p = sigmoid(X w).

    Steps:
        1. Compute p, then the curvature weights p * (1 - p), shape (n,).
        2. Scale each row of X by its weight without building an (n, n)
           matrix, then multiply by X transposed.
        3. Add ridge to the diagonal.
    """
    raise NotImplementedError("hessian")


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 2000,
    ridge: float = 0.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """Fit by fixed-step gradient descent, starting from w = 0.

    Record the loss before each update, so `history[0]` is the loss at the
    initial `w` and `history[-1]` is the loss just before the run stopped.
    Stop early, without taking that step, once the gradient norm falls below
    `tol`.

    Args:
        X: Design matrix, shape (n, d).
        y: Binary labels in {0, 1}, shape (n,).
        lr: Fixed step size.
        max_iter: Maximum number of steps.
        ridge: Nonnegative ridge coefficient.
        tol: Stop when the gradient norm drops below this value.

    Returns:
        A tuple `(w, history)`: the fitted coefficients, shape (d,), and the
        list of loss values recorded before each step.
    """
    raise NotImplementedError("fit_gradient_descent")


def fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    ridge: float = 1e-8,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Fit by Newton's method, starting from w = 0.

    Each step solves `H delta = g` for the Newton direction and updates
    `w -= delta`. Record the loss before each step, the same way
    `fit_gradient_descent` does, and stop early once the gradient norm falls
    below `tol`.

    Args:
        X: Design matrix, shape (n, d).
        y: Binary labels in {0, 1}, shape (n,).
        max_iter: Maximum number of steps.
        ridge: Nonnegative ridge coefficient, added to the Hessian's diagonal.
            Keep this at least as large as the default even on well-behaved
            data: it is what keeps the solve well posed.
        tol: Stop when the gradient norm drops below this value.

    Returns:
        A tuple `(w, history)`: the fitted coefficients, shape (d,), and the
        list of loss values recorded before each step.
    """
    raise NotImplementedError("fit_newton")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5, 0.8])
    y = (rng.uniform(size=n) < sigmoid(X @ w_true)).astype(float)

    w_newton, history_newton = fit_newton(X, y, max_iter=30)
    w_gd, history_gd = fit_gradient_descent(X, y, lr=0.01, max_iter=5000, tol=1e-7)

    print(f"newton: {len(history_newton)} iterations, final loss {history_newton[-1]:.4f}")
    print(f"gradient descent: {len(history_gd)} iterations, final loss {history_gd[-1]:.4f}")
    print(f"coefficients agree to {np.abs(w_newton - w_gd).max():.2e}")
