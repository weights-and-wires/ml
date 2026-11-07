"""Reference solution: gradient descent, curvature, step counts, and momentum on a bowl."""

from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_loss_and_grad(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    loss = 0.5 * (w @ A @ w) - b @ w
    grad = A @ w - b
    return float(loss), grad


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


def eigenvalue_range(A: np.ndarray) -> tuple[float, float]:
    eigs = np.linalg.eigvalsh(A)
    return float(eigs[0]), float(eigs[-1])


def condition_number(A: np.ndarray) -> float:
    mu, L = eigenvalue_range(A)
    return float(L / mu)


def axis_factors(A: np.ndarray, step: float) -> np.ndarray:
    eigs = np.linalg.eigvalsh(A)
    return np.abs(1.0 - step * eigs)


def steps_to_tolerance(path: np.ndarray, w_star: np.ndarray, tol: float) -> int:
    errors = np.linalg.norm(path - w_star, axis=1)
    hits = np.nonzero(errors < tol * errors[0])[0]
    return int(hits[0]) if hits.size else -1


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
    A = np.diag([1.0, 10.0])
    b = np.array([1.0, 10.0])
    w_star = np.array([1.0, 1.0])
    grad = lambda w: A @ w - b

    loss, g = quadratic_loss_and_grad(A, b, np.zeros(2))
    print(f"loss at (0, 0): {loss}, gradient: {g}")
    mu, L = eigenvalue_range(A)
    print(f"curvatures: mu = {mu}, L = {L}, safe step 1/L = {1 / L}, kappa = {condition_number(A):.0f}")
    for step in (0.1, 0.18, 0.3):
        path = gradient_descent(grad, np.zeros(2), step, 20)
        error = np.linalg.norm(path[-1] - w_star)
        print(f"step {step}: factors {axis_factors(A, step)}, error after 20 steps {error:.3g}")

    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    grad = lambda w: A @ w - A @ w_star
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
        print(f"{name:>17}: {steps_to_tolerance(path, w_star, 1e-6)} steps to reach one millionth of the starting miss")
