"""Reference solution for the logistic regression fitting exercise."""

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
    p = sigmoid(X @ w)
    return X.T @ (p - y) + ridge * w


def hessian(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    p = sigmoid(X @ w)
    bend = p * (1.0 - p)
    return (X * bend[:, None]).T @ X + ridge * np.eye(X.shape[1])


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


def train_accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    labels = (predict_proba(w, X) >= 0.5).astype(float)
    return float(np.mean(labels == y))


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

