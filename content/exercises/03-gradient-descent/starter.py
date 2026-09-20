"""Gradient descent, heavy ball, and Nesterov, all on a quadratic.

The objective throughout is

    f(w) = 0.5 * w^T A w - b^T w,       grad f(w) = A w - b,

with A symmetric positive definite. Its smallest and largest eigenvalues, mu
and L, are the strong-convexity and smoothness constants the lesson's rates
are stated in terms of, and kappa = L / mu is the condition number.

Fill in the five functions below. Each of gradient_descent, heavy_ball, and
nesterov returns the whole trajectory of iterates, not just the last one, so
that the checks and the lesson's plots can inspect convergence at every step.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_loss_and_grad(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate f(w) = 0.5 w^T A w - b^T w and its gradient at w.

    Args:
        A: Symmetric positive definite matrix, shape (d, d).
        b: Linear term, shape (d,).
        w: Point at which to evaluate, shape (d,).

    Returns:
        A tuple (loss, grad) with loss a float and grad shape (d,).
    """
    raise NotImplementedError("quadratic_loss_and_grad")


def gradient_descent(
    grad: Callable[[np.ndarray], np.ndarray], w0: np.ndarray, step: float, iters: int
) -> np.ndarray:
    """Run fixed-step gradient descent for `iters` steps.

    Args:
        grad: Function mapping a point w, shape (d,), to its gradient, shape (d,).
        w0: Starting point, shape (d,).
        step: Fixed step size.
        iters: Number of gradient steps to take.

    Returns:
        The trajectory of iterates, shape (iters + 1, d), with row 0 equal to
        w0 and row k equal to the iterate after k steps.
    """
    raise NotImplementedError("gradient_descent")


def exact_line_search_step(A: np.ndarray, grad_vec: np.ndarray) -> float:
    """Return the exact line-search step size for a quadratic with Hessian A.

    This is the alpha that minimises f(w - alpha * grad_vec) along the
    negative gradient direction, in closed form.

    Args:
        A: Symmetric positive definite matrix, shape (d, d).
        grad_vec: The gradient at the current point, shape (d,).

    Returns:
        The optimal step size, as a float.
    """
    raise NotImplementedError("exact_line_search_step")


def heavy_ball(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    """Run the heavy-ball method: w_{k+1} = w_k - step * grad(w_k) + momentum * (w_k - w_{k-1}).

    Treat w_{-1} as equal to w0, so the first update reduces to a plain
    gradient step.

    Args:
        grad: Function mapping a point w, shape (d,), to its gradient, shape (d,).
        w0: Starting point, shape (d,).
        step: Fixed step size.
        momentum: The heavy-ball momentum coefficient, in [0, 1).
        iters: Number of steps to take.

    Returns:
        The trajectory of iterates, shape (iters + 1, d).
    """
    raise NotImplementedError("heavy_ball")


def nesterov(
    grad: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    step: float,
    momentum: float,
    iters: int,
) -> np.ndarray:
    """Run Nesterov's accelerated gradient method.

    y_k = w_k + momentum * (w_k - w_{k-1})
    w_{k+1} = y_k - step * grad(y_k)

    Treat w_{-1} as equal to w0.

    Args:
        grad: Function mapping a point w, shape (d,), to its gradient, shape (d,).
        w0: Starting point, shape (d,).
        step: Fixed step size.
        momentum: The Nesterov momentum coefficient, in [0, 1).
        iters: Number of steps to take.

    Returns:
        The trajectory of iterates, shape (iters + 1, d).
    """
    raise NotImplementedError("nesterov")


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
