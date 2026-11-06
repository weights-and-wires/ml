"""Walk to the bottom of a bowl.

The bowl is f(w) = 0.5 * w^T A w - b^T w, with gradient A w - b. You build
the loss and gradient, a checker that estimates a gradient by nudging, and
gradient descent that returns the whole path.
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


def numerical_gradient(f: Callable[[np.ndarray], float], w: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Estimate the gradient of f at w by nudging one entry at a time.

    Entry i of the result is (f(w + h * e_i) - f(w - h * e_i)) / (2 * h),
    where e_i is 0 except for a 1 in entry i.

    Args:
        f: Function mapping a vector of shape (d,) to a float.
        w: Point at which to estimate the gradient, shape (d,). Do not change it.
        h: Size of the nudge.

    Returns:
        The estimated gradient, shape (d,).
    """
    raise NotImplementedError("numerical_gradient")


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
