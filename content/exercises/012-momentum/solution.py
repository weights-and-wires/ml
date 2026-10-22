"""Reference solution: heavy ball, Nesterov's look-ahead, and the best heavy-ball settings."""

from __future__ import annotations

from typing import Callable

import numpy as np


# Written in the chapter "Walking downhill: slope, gradient, and one step".
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


def heavy_ball(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    path = np.empty((iters + 1, len(w0)))
    path[0] = w0
    w_prev = np.array(w0, dtype=float)
    w = np.array(w0, dtype=float)
    for k in range(iters):
        w_next = w - step * grad(w) + momentum * (w - w_prev)
        w_prev, w = w, w_next
        path[k + 1] = w
    return path


def nesterov(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    path = np.empty((iters + 1, len(w0)))
    path[0] = w0
    w_prev = np.array(w0, dtype=float)
    w = np.array(w0, dtype=float)
    for k in range(iters):
        y = w + momentum * (w - w_prev)
        w_next = y - step * grad(y)
        w_prev, w = w, w_next
        path[k + 1] = w
    return path


def best_heavy_ball_params(mu: float, L: float) -> tuple[float, float]:
    root_L, root_mu = np.sqrt(L), np.sqrt(mu)
    step = 4.0 / (root_L + root_mu) ** 2
    momentum = ((root_L - root_mu) / (root_L + root_mu)) ** 2
    return float(step), float(momentum)


if __name__ == "__main__":
    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = w_star + np.array([1.0, 1.0])

    step, momentum = best_heavy_ball_params(mu, L)
    nesterov_momentum = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    walks = {
        "gradient descent": gradient_descent(grad, w0, 1.0 / L, 1500),
        "heavy ball": heavy_ball(grad, w0, step, momentum, 1500),
        "nesterov": nesterov(grad, w0, 1.0 / L, nesterov_momentum, 1500),
    }
    print(f"heavy ball settings for kappa = {L / mu:.0f}: step {step:.4f}, momentum {momentum:.4f}")
    for name, path in walks.items():
        errors = np.linalg.norm(path - w_star, axis=1)
        hit = np.nonzero(errors < 1e-6 * errors[0])[0]
        steps = int(hit[0]) if hit.size else None
        print(f"{name:>17}: {steps} steps to reach one millionth of the starting miss")
