"""Measure how slow a long valley is.

At step 1/L, gradient descent on a bowl shrinks its miss along the flat axis
by 1 - mu / L = 1 - 1 / kappa per step, so the number of steps grows in
proportion to the condition number kappa = L / mu. You build the tools that
measure kappa, the shrink ratio of a walk, and the number of steps it takes.
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


# Written in the chapter "How big a step: curvature and the 2/L limit".
def eigenvalue_range(A: np.ndarray) -> tuple[float, float]:
    eigs = np.linalg.eigvalsh(A)
    return float(eigs[0]), float(eigs[-1])


def condition_number(A: np.ndarray) -> float:
    """Return kappa = L / mu, the sharpest curvature over the flattest.

    Args:
        A: Symmetric positive definite matrix, shape (d, d).

    Returns:
        The condition number, as a float.
    """
    raise NotImplementedError("condition_number")


def contraction_ratios(path: np.ndarray, w_star: np.ndarray) -> np.ndarray:
    """Return the ratio of each error to the error before it.

    The error at row k is the length of path[k] - w_star.

    Args:
        path: The walk, shape (iters + 1, d).
        w_star: The bottom of the bowl, shape (d,).

    Returns:
        An array of shape (iters,) whose entry k is error[k + 1] / error[k].
    """
    raise NotImplementedError("contraction_ratios")


def steps_to_tolerance(path: np.ndarray, w_star: np.ndarray, tol: float) -> int:
    """Return the first row whose error is below tol times the starting error.

    Args:
        path: The walk, shape (iters + 1, d).
        w_star: The bottom of the bowl, shape (d,).
        tol: Fraction of the starting error to reach, for example 1e-3.

    Returns:
        The index of the first row with error < tol * error[0], or -1 if no
        row gets there.
    """
    raise NotImplementedError("steps_to_tolerance")


if __name__ == "__main__":
    w_star = np.array([1.0, 1.0])
    for curvature in (10.0, 100.0):
        A = np.diag([1.0, curvature])
        b = A @ w_star
        grad = lambda w: A @ w - b
        kappa = condition_number(A)
        path = gradient_descent(grad, np.array([0.0, 1.0]), 1.0 / curvature, 1000)
        ratios = contraction_ratios(path, w_star)
        steps = steps_to_tolerance(path, w_star, 1e-3)
        print(f"kappa = {kappa:.0f}: shrink {ratios[5]:.4f} per step, {steps} steps to reach one thousandth")
