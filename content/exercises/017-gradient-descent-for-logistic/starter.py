"""Fit a logistic regression by gradient descent.

You write three functions, in this order:

1. `gradient`: the slope of the log loss, shape (d,).
2. `fit_gradient_descent`: fit by small fixed steps, recording the loss.
3. `train_accuracy`: the fraction of rows labelled correctly.

`sigmoid`, `predict_proba`, `softplus`, and `log_loss` are provided, because
you wrote them in the earlier chapters.

Shapes used everywhere: X is (n, d), one row per example; y is (n,) with
labels 0 or 1; w is (d,). The loss is a sum over examples, not a mean:

    L(w) = sum_i [softplus(z_i) - y_i * z_i] + (ridge / 2) * ||w||^2,
    where z = X @ w.
"""

import numpy as np


# Provided: you wrote sigmoid in the chapter "From a score to a probability".
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


# Provided: you wrote predict_proba in the chapter "From a score to a probability".
def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


# Provided: you wrote softplus and log_loss in the chapter "How wrong is a probability? The log loss".
def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    z = X @ w
    penalty = 0.5 * ridge * float(np.dot(w, w))
    return float(np.sum(softplus(z) - y * z)) + penalty


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Return the gradient of `log_loss` with respect to `w`.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        ridge: Strength of the ridge fee.

    Returns:
        An array of shape (d,): X.T @ (sigmoid(X @ w) - y) + ridge * w.
    """
    raise NotImplementedError("write gradient")


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 2000,
    ridge: float = 0.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """Fit the weights by gradient descent, starting from zeros.

    Args:
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        lr: Step size.
        max_iter: Maximum number of loop iterations.
        ridge: Strength of the ridge fee.
        tol: Stop when the gradient's length is below this value.

    Returns:
        A pair (w, history). `history` holds the loss recorded at the start of
        every iteration, before the step, so history[0] is the loss at zero
        weights. When the gradient is already shorter than `tol`, stop
        without taking the step.
    """
    raise NotImplementedError("write fit_gradient_descent")


def train_accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Return the fraction of rows whose predicted label equals y.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).

    Returns:
        A float between 0 and 1. A row is labelled 1 when its probability is
        at least 0.5.
    """
    raise NotImplementedError("write train_accuracy")


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X2 = np.array([[1.0, 0.0], [1.0, 2.0]])
    y2 = np.array([1.0, 0.0])
    w2 = np.array([0.5, -1.0])
    print("gradient on the two emails:", np.round(gradient(w2, X2, y2), 4))
    step = w2 - 1.0 * gradient(w2, X2, y2)
    print("after one step of size 1  :", np.round(step, 4), "loss", round(log_loss(step, X2, y2), 4))

    # A 60-email inbox: a bias column and the number of earlier replies.
    rng = np.random.default_rng(3)
    replies = rng.integers(0, 6, size=60).astype(float)
    X = np.column_stack([np.ones(60), replies])
    y = (rng.uniform(size=60) < sigmoid(X @ np.array([1.0, -0.9]))).astype(float)
    w, history = fit_gradient_descent(X, y, lr=0.02, max_iter=2000, tol=1e-6)
    print(f"inbox fit: {len(history)} steps, loss {history[0]:.2f} -> {history[-1]:.2f}, "
          f"accuracy {train_accuracy(w, X, y):.3f}")

    # Perfect split: the weights keep growing without a ridge fee.
    for ridge in (0.0, 1.0):
        w_sep, _ = fit_gradient_descent(X2, y2, lr=0.5, max_iter=3000, ridge=ridge, tol=0.0)
        print(f"two emails, ridge {ridge}: |w| = {np.linalg.norm(w_sep):.2f}")
