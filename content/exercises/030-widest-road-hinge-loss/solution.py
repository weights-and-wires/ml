"""Reference solution: the widest road, trained with the hinge loss."""

import numpy as np


def functional_margins(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y * (X @ w + b)


def hinge_loss_and_subgradient(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, C: float):
    z = functional_margins(w, b, X, y)
    loss = 0.5 * float(w @ w) + C * float(np.sum(np.maximum(0.0, 1.0 - z)))
    active = z < 1.0
    grad_w = w - C * (y[active] @ X[active])
    grad_b = -C * float(np.sum(y[active]))
    return loss, grad_w, grad_b


def fit_subgradient(X: np.ndarray, y: np.ndarray, C: float, lr: float = 0.3, steps: int = 3000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for t in range(steps):
        _, grad_w, grad_b = hinge_loss_and_subgradient(w, b, X, y, C)
        step = lr / (1.0 + lr * t)
        w = w - step * grad_w
        b = b - step * grad_b
    return w, b


def houses():
    """Twelve houses on a map, in units of 100 m. Circles are -1 and squares are +1."""
    X = np.array(
        [
            [3.0, 4.5], [2.7, 3.4], [2.5, 3.3], [3.1, 5.4], [2.5, 3.6], [3.5, 4.5],
            [8.1, 1.6], [8.0, 3.0], [6.7, 2.0], [6.1, 1.2], [6.2, 2.2], [6.7, 2.6],
        ]
    )
    y = np.array([-1.0] * 6 + [1.0] * 6)
    return X, y


if __name__ == "__main__":
    X, y = houses()
    C = 1.0
    w, b = fit_subgradient(X, y, C)
    z = functional_margins(w, b, X, y)
    loss, _, _ = hinge_loss_and_subgradient(w, b, X, y, C)
    print(f"w = {w}, b = {b:.3f}")
    print(f"smallest margin z = {z.min():.3f} (a value near 1 means a point rests on the road edge)")
    print(f"road half-width 1/||w|| = {1.0 / np.linalg.norm(w):.3f}")
    print(f"objective J = {loss:.4f}")
