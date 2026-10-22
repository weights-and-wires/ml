"""Reference solution: loss and gradient of a bowl, a nudge checker, and gradient descent."""

from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_loss_and_grad(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    loss = 0.5 * (w @ A @ w) - b @ w
    grad = A @ w - b
    return float(loss), grad


def numerical_gradient(f: Callable[[np.ndarray], float], w: np.ndarray, h: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        e = np.zeros_like(w, dtype=float)
        e[i] = 1.0
        grad[i] = (f(w + h * e) - f(w - h * e)) / (2 * h)
    return grad


def gradient_descent(
    grad: Callable[[np.ndarray], np.ndarray], w0: np.ndarray, step: float, iters: int
) -> np.ndarray:
    path = np.empty((iters + 1, len(w0)))
    path[0] = w0
    w = np.array(w0, dtype=float)
    for k in range(iters):
        w = w - step * grad(w)
        path[k + 1] = w
    return path


if __name__ == "__main__":
    A = np.diag([1.0, 10.0])
    b = np.array([1.0, 10.0])
    w = np.array([0.0, 0.0])

    loss, grad = quadratic_loss_and_grad(A, b, w)
    print(f"loss at (0, 0): {loss}, gradient: {grad}")

    f = lambda v: quadratic_loss_and_grad(A, b, v)[0]
    print(f"nudged gradient at (0, 0): {numerical_gradient(f, w)}")

    path = gradient_descent(lambda v: quadratic_loss_and_grad(A, b, v)[1], w, 0.1, 5)
    for k, point in enumerate(path):
        print(f"step {k}: w = {point}, loss = {f(point):.4f}")
