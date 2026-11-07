"""Walk down a long valley, with and without momentum.

The bowl is f(w) = 0.5 * w^T A w - b^T w, with gradient A w - b. You build the
loss and gradient, gradient descent, the curvature readings that predict the
safe step and the step count, and then heavy-ball and Nesterov momentum.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_loss_and_grad(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate f(w) = 0.5 w^T A w - b^T w and its gradient at w.

    Args:
        A: Symmetric matrix, shape (d, d).
        b: Linear term, shape (d,).
        w: Point at which to evaluate, shape (d,).

    Returns:
        A tuple (loss, grad). The loss is a Python float and grad has shape (d,).
    """
    raise NotImplementedError("quadratic_loss_and_grad")


def gradient_descent(
    grad: Callable[[np.ndarray], np.ndarray], w0: np.ndarray, step: float, iters: int
) -> np.ndarray:
    """Run fixed-step gradient descent for `iters` steps.

    Args:
        grad: Function mapping a point w, shape (d,), to its gradient, shape (d,).
        w0: Starting point, shape (d,). Do not change it.
        step: Fixed step size.
        iters: Number of steps to take.

    Returns:
        The path, shape (iters + 1, d). Row 0 is w0 and row k is the point
        after k steps.
    """
    raise NotImplementedError("gradient_descent")


def eigenvalue_range(A: np.ndarray) -> tuple[float, float]:
    """Return the smallest and largest curvature of a symmetric matrix.

    Args:
        A: Symmetric matrix, shape (d, d).

    Returns:
        A tuple (mu, L) of two Python floats: the smallest and largest eigenvalue.
    """
    raise NotImplementedError("eigenvalue_range")


def condition_number(A: np.ndarray) -> float:
    """Return kappa = L / mu, the sharpest curvature over the flattest.

    Args:
        A: Symmetric positive definite matrix, shape (d, d).

    Returns:
        The condition number, as a float.
    """
    raise NotImplementedError("condition_number")


def axis_factors(A: np.ndarray, step: float) -> np.ndarray:
    """Return the size of the shrink factor on each axis for a step size.

    For the axis with eigenvalue lambda the factor is |1 - step * lambda|. A
    factor above 1 means the distance along that axis grows.

    Args:
        A: Symmetric matrix, shape (d, d).
        step: Step size.

    Returns:
        An array of shape (d,), one entry per eigenvalue, in ascending order of eigenvalue.
    """
    raise NotImplementedError("axis_factors")


def steps_to_tolerance(path: np.ndarray, w_star: np.ndarray, tol: float) -> int:
    """Return the first row whose error is below tol times the starting error.

    The error at row k is the length of path[k] - w_star.

    Args:
        path: The walk, shape (iters + 1, d).
        w_star: The bottom of the bowl, shape (d,).
        tol: Fraction of the starting error to reach, for example 1e-3.

    Returns:
        The index of the first row with error < tol * error[0], or -1 if no
        row gets there.
    """
    raise NotImplementedError("steps_to_tolerance")


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
