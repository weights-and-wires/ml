"""Reference solution for the Newton's method exercise."""

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
    p = sigmoid(X @ w)
    bend = p * (1.0 - p)
    return (X * bend[:, None]).T @ X + ridge * np.eye(X.shape[1])


def newton_step(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    return np.linalg.solve(hessian(w, X, y, ridge), gradient(w, X, y, ridge))


def fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    ridge: float = 1e-8,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    w = np.zeros(X.shape[1])
    history: list[float] = []
    for _ in range(max_iter):
        history.append(log_loss(w, X, y, ridge))
        if np.linalg.norm(gradient(w, X, y, ridge)) < tol:
            break
        w = w - newton_step(w, X, y, ridge)
    return w, history


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
