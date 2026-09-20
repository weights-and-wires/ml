"""Reference solution for gradient descent on a quadratic."""

from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_loss_and_grad(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    loss = 0.5 * w @ A @ w - b @ w
    grad = A @ w - b
    return float(loss), grad


def gradient_descent(
    grad: Callable[[np.ndarray], np.ndarray], w0: np.ndarray, step: float, iters: int
) -> np.ndarray:
    d = w0.shape[0]
    trajectory = np.empty((iters + 1, d))
    trajectory[0] = w0
    w = w0.copy()
    for k in range(iters):
        w = w - step * grad(w)
        trajectory[k + 1] = w
    return trajectory


def exact_line_search_step(A: np.ndarray, grad_vec: np.ndarray) -> float:
    return float((grad_vec @ grad_vec) / (grad_vec @ A @ grad_vec))


def heavy_ball(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    d = w0.shape[0]
    trajectory = np.empty((iters + 1, d))
    trajectory[0] = w0
    w_prev = w0.copy()
    w = w0.copy()
    for k in range(iters):
        w_next = w - step * grad(w) + momentum * (w - w_prev)
        w_prev, w = w, w_next
        trajectory[k + 1] = w
    return trajectory


def nesterov(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    d = w0.shape[0]
    trajectory = np.empty((iters + 1, d))
    trajectory[0] = w0
    w_prev = w0.copy()
    w = w0.copy()
    for k in range(iters):
        y = w + momentum * (w - w_prev)
        w_next = y - step * grad(y)
        w_prev, w = w, w_next
        trajectory[k + 1] = w
    return trajectory


if __name__ == "__main__":
    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b

    w0 = w_star + np.array([1.0, 1.0])
    kappa = L / mu

    gd = gradient_descent(grad, w0, 1.0 / L, 200)
    hb_step = (2.0 / (np.sqrt(L) + np.sqrt(mu))) ** 2
    hb_momentum = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    hb = heavy_ball(grad, w0, hb_step, hb_momentum, 200)
    nag_momentum = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    nag = nesterov(grad, w0, 1.0 / L, nag_momentum, 200)

    print(f"condition number: {kappa:.1f}")
    for name, trajectory in [("gradient descent", gd), ("heavy ball", hb), ("nesterov", nag)]:
        errors = np.linalg.norm(trajectory - w_star, axis=1)
        print(f"{name:>17}: error after 20 steps {errors[20]:.3e}, after 200 steps {errors[-1]:.3e}")
