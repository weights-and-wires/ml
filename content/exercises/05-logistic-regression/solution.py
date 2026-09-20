"""Reference solution for the logistic regression exercise."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _softplus(z: np.ndarray) -> np.ndarray:
    # log(1 + exp(z)) through log-sum-exp: factor out the larger of the two
    # terms in {0, z} so the remaining exponential never overflows.
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    z = X @ w
    penalty = 0.5 * ridge * float(np.dot(w, w))
    return float(np.sum(_softplus(z) - y * z)) + penalty


def gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    z = X @ w
    p = sigmoid(z)
    return X.T @ (p - y) + ridge * w


def hessian(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    z = X @ w
    p = sigmoid(z)
    d = p * (1.0 - p)
    return (X * d[:, None]).T @ X + ridge * np.eye(X.shape[1])


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
        g = gradient(w, X, y, ridge)
        if np.linalg.norm(g) < tol:
            break
        H = hessian(w, X, y, ridge)
        w = w - np.linalg.solve(H, g)
    return w, history


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
