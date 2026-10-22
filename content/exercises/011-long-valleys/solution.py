"""Reference solution: condition number, shrink ratios, and step counts."""

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
