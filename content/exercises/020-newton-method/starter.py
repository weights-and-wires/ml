"""Fit a logistic regression with Newton's method.

You write three functions, in this order:

1. `hessian`: second derivatives of the loss, shape (d, d).
2. `newton_step`: the vector delta that solves H delta = g.
3. `fit_newton`: fit by repeating w = w - delta from zero weights.

`sigmoid`, `softplus`, `log_loss`, `gradient`, and `fit_gradient_descent` are
provided, because you wrote them in the earlier chapters.

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


# Provided: you wrote softplus and log_loss in the chapter "How wrong is a probability? The log loss".
def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    z = X @ w
    penalty = 0.5 * ridge * float(np.dot(w, w))
    return float(np.sum(softplus(z) - y * z)) + penalty


# Provided: you wrote gradient and fit_gradient_descent in the chapter "Learning the weights: gradient and descent".
def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    p = sigmoid(X @ w)
    return X.T @ (p - y) + ridge * w


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 2000,
    ridge: float = 0.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    w = np.zeros(X.shape[1])
    history: list[float] = []
    for _ in range(max_iter):
        history.append(log_loss(w, X, y, ridge))
        g = gradient(w, X, y, ridge)
        if np.linalg.norm(g) < tol:
            break
        w = w - lr * g
    return w, history


def hessian(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Return the Hessian of `log_loss` with respect to `w`.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,). The Hessian does not depend on it.
        ridge: Strength of the ridge fee.

    Returns:
        A symmetric array of shape (d, d): X.T @ D @ X + ridge * I, where D is
        diagonal with entries p * (1 - p) and p = sigmoid(X @ w). Scale the
        rows of X by p * (1 - p) instead of building the (n, n) matrix D.
    """
    raise NotImplementedError("write hessian")


def newton_step(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Return delta, the vector to subtract from `w` for one Newton step.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        ridge: Strength of the ridge fee.

    Returns:
        delta of shape (d,), the solution of H delta = g, where H is the
        Hessian and g is the gradient at `w`. Use np.linalg.solve, not an
        inverse. The function does not subtract anything itself.
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
        max_iter: Maximum number of loop passes.
        ridge: Strength of the ridge fee. The small default keeps the solve
            valid when the Hessian is nearly singular.
        tol: Stop when the gradient's length is below this value.

    Returns:
        A pair (w, history). `history` holds the loss recorded at the start of
        every pass, before the step, so history[0] is the loss at zero
        weights. When the gradient is already shorter than `tol`, stop
        without taking the step.
    """
    raise NotImplementedError("write fit_newton")


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X2 = np.array([[1.0, 0.0], [1.0, 2.0]])
    y2 = np.array([1.0, 0.0])
    w2 = np.array([0.5, -1.0])
    print("hessian at (0.5, -1):\n", np.round(hessian(w2, X2, y2), 4))
    delta = newton_step(w2, X2, y2)
    print("newton step delta   :", np.round(delta, 4), "-> new weights", np.round(w2 - delta, 4))
    print("step from zero      :", np.round(newton_step(np.zeros(2), X2, y2), 4))

    # A 60-email inbox: Newton against gradient descent.
    rng = np.random.default_rng(3)
    replies = rng.integers(0, 6, size=60).astype(float)
    X = np.column_stack([np.ones(60), replies])
    y = (rng.uniform(size=60) < sigmoid(X @ np.array([1.0, -0.9]))).astype(float)
    w_newton, h_newton = fit_newton(X, y, ridge=0.0)
    w_gd, h_gd = fit_gradient_descent(X, y, lr=0.02, max_iter=2000, tol=1e-6)
    print(f"newton: {len(h_newton)} passes, loss {h_newton[-1]:.4f}")
    print(f"gradient descent: {len(h_gd)} passes, loss {h_gd[-1]:.4f}")
    print("weights agree to", f"{np.abs(w_newton - w_gd).max():.1e}")

    # Perfect split: no ridge marches off, a real ridge fee stops.
    for ridge in (1e-8, 1.0):
        w_sep, h_sep = fit_newton(X2, y2, ridge=ridge)
        print(f"two emails, ridge {ridge:g}: {len(h_sep)} passes, |w| = {np.linalg.norm(w_sep):.2f}")
