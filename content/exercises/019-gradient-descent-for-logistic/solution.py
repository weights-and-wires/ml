"""Reference solution for the gradient descent exercise."""

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
