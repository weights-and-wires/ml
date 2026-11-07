"""Fit a logistic regression with gradient descent and Newton's method.

You write six functions, in this order:

1. `gradient`: the slope of the log loss, shape (d,).
2. `hessian`: the second derivatives of the loss, shape (d, d).
3. `fit_gradient_descent`: fit by small fixed steps, recording the loss.
4. `newton_step`: the vector delta that solves H delta = g.
5. `fit_newton`: fit by repeating w = w - delta from zero weights.
6. `train_accuracy`: the fraction of rows labelled correctly.

`sigmoid`, `predict_proba`, `softplus`, and `log_loss` are provided, because
you wrote them in the previous chapter.

Shapes used everywhere: X is (n, d), one row per example; y is (n,) with
labels 0 or 1; w is (d,). The loss is a sum over examples, not a mean:

    L(w) = sum_i [softplus(z_i) - y_i * z_i] + (ridge / 2) * ||w||^2,
    where z = X @ w.
"""

import numpy as np


# Provided: you wrote these in the chapter "From a score to a probability, and how wrong it is".
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    z = X @ w
    penalty = 0.5 * ridge * float(np.dot(w, w))
    return float(np.sum(softplus(z) - y * z)) + penalty


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Return the gradient of the log loss with respect to the weights.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        ridge: Strength of the ridge fee.

    Returns:
        X^T (sigmoid(X w) - y) + ridge * w, shape (d,).
    """
    raise NotImplementedError("write gradient")


def hessian(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Return the Hessian of the log loss with respect to the weights.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,). The Hessian does not depend on them.
        ridge: Strength of the ridge fee.

    Returns:
        X^T D X + ridge * I, shape (d, d), where D is the diagonal matrix of
        p_i (1 - p_i) and p = sigmoid(X w). Scale the rows of X by the bends
        instead of building the n by n matrix D.
    """
    raise NotImplementedError("write hessian")


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
        max_iter: Largest number of passes.
        ridge: Strength of the ridge fee.
        tol: Stop when the gradient's norm is below this.

    Returns:
        The weights, shape (d,), and a list with the loss recorded before each
        step, so history[0] is the loss at zero weights.

    Steps:
        1. Record the loss, compute the gradient, and stop if its norm is
           below `tol`.
        2. Otherwise step against the gradient and repeat.
    """
    raise NotImplementedError("write fit_gradient_descent")


def newton_step(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Return delta, the vector to subtract from w, as the solution of H delta = g.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        ridge: Strength of the ridge fee.

    Returns:
        delta, shape (d,). Use `np.linalg.solve`, not an explicit inverse. It
        does not subtract anything itself.
    """
    raise NotImplementedError("write newton_step")


def fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    ridge: float = 1e-8,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Fit the weights with Newton's method, starting from zeros.

    Args:
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        max_iter: Largest number of passes.
        ridge: Strength of the ridge fee. The small default keeps the solve
            valid when two columns are copies of each other.
        tol: Stop when the gradient's norm is below this.

    Returns:
        The weights, shape (d,), and a list with the loss recorded before each
        step, so history[0] is the loss at zero weights.
    """
    raise NotImplementedError("write fit_newton")


def train_accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Return the fraction of rows whose predicted label equals the true label.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).

    Returns:
        The mean of (predict_proba(w, X) >= 0.5) == y, as a float.
    """
    raise NotImplementedError("write train_accuracy")


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X2 = np.array([[1.0, 0.0], [1.0, 2.0]])
    y2 = np.array([1.0, 0.0])
    w2 = np.array([0.5, -1.0])
    print("gradient on the two emails:", np.round(gradient(w2, X2, y2), 4))
    step = w2 - 1.0 * gradient(w2, X2, y2)
    print("after one gradient step   :", np.round(step, 4), "loss", round(log_loss(step, X2, y2), 4))
    print("hessian at (0.5, -1):\n", np.round(hessian(w2, X2, y2), 4))
    delta = newton_step(w2, X2, y2)
    print("newton step               :", np.round(w2 - delta, 4), "loss", round(log_loss(w2 - delta, X2, y2), 4))

    # A 60-email inbox: a bias column and the number of earlier replies.
    rng = np.random.default_rng(3)
    replies = rng.integers(0, 6, size=60).astype(float)
    X = np.column_stack([np.ones(60), replies])
    y = (rng.uniform(size=60) < sigmoid(X @ np.array([1.0, -0.9]))).astype(float)
    w_gd, h_gd = fit_gradient_descent(X, y, lr=0.02, max_iter=2000, tol=1e-6)
    w_newton, h_newton = fit_newton(X, y, ridge=0.0)
    print(f"gradient descent: {len(h_gd)} passes, loss {h_gd[-1]:.4f}, accuracy {train_accuracy(w_gd, X, y):.3f}")
    print(f"newton          : {len(h_newton)} passes, loss {h_newton[-1]:.4f}")
    print("weights agree to", f"{np.abs(w_newton - w_gd).max():.1e}")

    # Perfect split: the weights grow without a real ridge fee.
    for ridge in (1e-8, 1.0):
        w_sep, h_sep = fit_newton(X2, y2, ridge=ridge)
        print(f"two emails, newton, ridge {ridge:g}: {len(h_sep)} passes, |w| = {np.linalg.norm(w_sep):.2f}")

