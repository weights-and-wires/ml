"""Find the safe step, then measure how slow a long valley is.

On a bowl f(w) = 0.5 * w^T A w - b^T w, one step of size alpha multiplies the
distance to the bottom along each axis by 1 - alpha * lambda, where lambda is
the curvature along that axis (an eigenvalue of A). At step 1/L the flat axis
shrinks by 1 - 1 / kappa per step, where kappa = L / mu, so the number of steps
grows in proportion to kappa. You build the pieces that read the curvatures,
predict the shrink factors, pick the best step along a line, and measure kappa,
the shrink ratio of a walk, and the number of steps it takes.
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


def eigenvalue_range(A: np.ndarray) -> tuple[float, float]:
    """Return the smallest and largest curvature of a symmetric matrix.

    Args:
        A: Symmetric matrix, shape (d, d).

    Returns:
        A tuple (mu, L) of two Python floats: the smallest and largest eigenvalue.
    """
    raise NotImplementedError("eigenvalue_range")


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


def exact_line_search_step(A: np.ndarray, grad_vec: np.ndarray) -> float:
    """Return the step that makes the height lowest along the downhill line.

    This is the alpha that minimises f(w - alpha * grad_vec) in closed form.

    Args:
        A: Symmetric positive definite matrix, shape (d, d).
        grad_vec: The gradient at the current point, shape (d,).

    Returns:
        The best step size, as a float.
    """
    raise NotImplementedError("exact_line_search_step")


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
    A = np.diag([1.0, 10.0])
    b = np.array([1.0, 10.0])
    grad = lambda w: A @ w - b
    w_star = np.array([1.0, 1.0])

    mu, L = eigenvalue_range(A)
    print(f"curvatures: mu = {mu}, L = {L}, safe step 1/L = {1 / L}, kappa = {condition_number(A):.0f}")

    for step in (0.1, 0.18, 0.3):
        path = gradient_descent(grad, np.zeros(2), step, 20)
        error = np.linalg.norm(path[-1] - w_star)
        print(f"step {step}: factors {axis_factors(A, step)}, error after 20 steps {error:.3g}")

    g = grad(np.zeros(2))
    print(f"best step along the downhill line at (0, 0): {exact_line_search_step(A, g):.4f}")

    for curvature in (10.0, 100.0):
        A = np.diag([1.0, curvature])
        grad = lambda w: A @ w - A @ w_star
        path = gradient_descent(grad, np.array([0.0, 1.0]), 1.0 / curvature, 1000)
        ratios = contraction_ratios(path, w_star)
        steps = steps_to_tolerance(path, w_star, 1e-3)
        print(f"kappa = {curvature:.0f}: shrink {ratios[5]:.4f} per step, {steps} steps to reach one thousandth")
