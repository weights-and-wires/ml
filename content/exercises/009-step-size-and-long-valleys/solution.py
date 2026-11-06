"""Reference solution: curvatures, shrink factors, line search, condition number, and step counts."""

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
    eigs = np.linalg.eigvalsh(A)
    return float(eigs[0]), float(eigs[-1])


def axis_factors(A: np.ndarray, step: float) -> np.ndarray:
    eigs = np.linalg.eigvalsh(A)
    return np.abs(1.0 - step * eigs)


def exact_line_search_step(A: np.ndarray, grad_vec: np.ndarray) -> float:
    return float((grad_vec @ grad_vec) / (grad_vec @ A @ grad_vec))


def condition_number(A: np.ndarray) -> float:
    mu, L = eigenvalue_range(A)
    return float(L / mu)


def contraction_ratios(path: np.ndarray, w_star: np.ndarray) -> np.ndarray:
    errors = np.linalg.norm(path - w_star, axis=1)
    return errors[1:] / errors[:-1]


def steps_to_tolerance(path: np.ndarray, w_star: np.ndarray, tol: float) -> int:
    errors = np.linalg.norm(path - w_star, axis=1)
    hits = np.nonzero(errors < tol * errors[0])[0]
    return int(hits[0]) if hits.size else -1


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
