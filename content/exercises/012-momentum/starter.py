"""Roll a heavy ball down a long valley.

Heavy ball adds a fraction of the last step to each gradient step. Nesterov's
method measures the gradient at the point momentum is carrying you to. Both
need about sqrt(kappa) steps on a bowl, where plain descent needs about kappa.
"""

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
    """Run heavy ball: w_next = w - step * grad(w) + momentum * (w - w_prev).

    Treat the point before w0 as w0 itself, so the first step is a plain gradient step.

    Args:
        grad: Function mapping a point w, shape (d,), to its gradient, shape (d,).
        w0: Starting point, shape (d,). Do not change it.
        step: Step size.
        momentum: Fraction of the last step to keep, in [0, 1).
        iters: Number of steps to take.

    Returns:
        The path, shape (iters + 1, d), with row 0 equal to w0.
    """
    raise NotImplementedError("heavy_ball")


def nesterov(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    """Run Nesterov's method: y = w + momentum * (w - w_prev), then w_next = y - step * grad(y).

    Treat the point before w0 as w0 itself.

    Args:
        grad: Function mapping a point w, shape (d,), to its gradient, shape (d,).
        w0: Starting point, shape (d,). Do not change it.
        step: Step size.
        momentum: Fraction of the last step to look ahead by, in [0, 1).
        iters: Number of steps to take.

    Returns:
        The path, shape (iters + 1, d), with row 0 equal to w0.
    """
    raise NotImplementedError("nesterov")


def best_heavy_ball_params(mu: float, L: float) -> tuple[float, float]:
    """Return the fastest (step, momentum) for heavy ball on a bowl with curvatures in [mu, L].

    Args:
        mu: Smallest curvature (eigenvalue).
        L: Largest curvature (eigenvalue).

    Returns:
        A tuple (step, momentum) of two floats.
    """
    raise NotImplementedError("best_heavy_ball_params")


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
