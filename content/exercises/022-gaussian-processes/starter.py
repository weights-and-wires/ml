"""Gaussian processes: prior draws, the posterior, and the log marginal likelihood.

Fill in the six functions below, in order. The first is the RBF kernel with a
length scale. The next two draw functions from the prior and condition on data.
The last three score a length scale by the evidence and pick the best one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]

JITTER = 1e-8


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
    """Return the RBF kernel matrix exp(-||x - x'||^2 / (2 length_scale^2)).

    Args:
        X1: Points, shape (n1, d).
        X2: Points, shape (n2, d).
        length_scale: Distance over which the similarity falls to exp(-1/2), > 0.

    Returns:
        Similarities, shape (n1, n2). Every diagonal entry of k(X, X) is 1.
    """
    raise NotImplementedError("write rbf_kernel")


def gp_prior_samples(X: np.ndarray, kernel: Kernel, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n random functions from a zero-mean GP prior, evaluated at X.

    Factor K + JITTER * I as L L^T, draw standard normals z, and return L z.
    The jitter 1e-8 keeps the factorisation from failing on a Gram matrix
    whose smallest eigenvalue rounds to a tiny negative number.

    Args:
        X: Input points, shape (m, d).
        kernel: Function that maps (A, B) to the (len(A), len(B)) Gram matrix.
        n: Number of functions to draw.
        rng: Source of randomness. Draw one array of shape (n, m) from it.

    Returns:
        Samples, shape (n, m). Row i is one function evaluated at the m points.
    """
    raise NotImplementedError("write gp_prior_samples")


def gp_posterior(
    X: np.ndarray, y: np.ndarray, Xs: np.ndarray, kernel: Kernel, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the posterior mean and variance of the function at the points Xs.

    The model is y = f(x) + e with f drawn from a zero-mean GP and e noise of
    variance `noise`. With K = k(X, X), Ks = k(Xs, X), and A = K + noise I,

        mean     = Ks A^{-1} y
        variance = k(x*, x*) - k(x*)^T A^{-1} k(x*)   for each row x* of Xs.

    The variance is for the function f, not for a new noisy target, so add
    `noise` to it for the target.

    Args:
        X: Training inputs, shape (n, d).
        y: Training targets, shape (n,).
        Xs: Query inputs, shape (m, d).
        kernel: Function that maps (A, B) to the (len(A), len(B)) Gram matrix.
        noise: Noise variance, > 0.

    Returns:
        (mean, variance), each of shape (m,).
    """
    raise NotImplementedError("write gp_posterior")


def log_marginal_likelihood(X: np.ndarray, y: np.ndarray, length_scale: float, noise: float) -> float:
    """Return log p(y | X) for an RBF GP with the given length scale and noise.

    With A = K + noise I, this is
    -0.5 y^T A^{-1} y - 0.5 log det A - 0.5 n log(2 pi).
    Use a Cholesky factor L of A: log det A = 2 * sum(log(diag(L))).

    Args:
        X: Training inputs, shape (n, d).
        y: Training targets, shape (n,).
        length_scale: RBF length scale, > 0.
        noise: Noise variance, > 0.

    Returns:
        The log marginal likelihood as a Python float.
    """
    raise NotImplementedError("write log_marginal_likelihood")


def lml_grad_log_length_scale(X: np.ndarray, y: np.ndarray, length_scale: float, noise: float) -> float:
    """Return d log p(y | X) / d log(length_scale).

    With A = K + noise I and alpha = A^{-1} y, the slope is
    0.5 * trace((alpha alpha^T - A^{-1}) dK), where for the RBF kernel
    dK = K * D2 / length_scale^2 entry by entry, and D2 holds the squared
    distances between the training points.

    Args:
        X: Training inputs, shape (n, d).
        y: Training targets, shape (n,).
        length_scale: RBF length scale, > 0.
        noise: Noise variance, > 0.

    Returns:
        The slope as a Python float.
    """
    raise NotImplementedError("write lml_grad_log_length_scale")


def best_length_scale(X: np.ndarray, y: np.ndarray, noise: float, grid: np.ndarray) -> float:
    """Return the length scale in `grid` with the highest log marginal likelihood.

    Args:
        X: Training inputs, shape (n, d).
        y: Training targets, shape (n,).
        noise: Noise variance, > 0.
        grid: Candidate length scales, shape (g,).

    Returns:
        One entry of `grid`, as a Python float.
    """
    raise NotImplementedError("write best_length_scale")


if __name__ == "__main__":
    X = np.array([0.51, 2.37, 3.21, 3.82, 5.93, 6.13])[:, None]
    y = np.array([0.36, 0.73, -0.39, -0.58, -0.30, 0.17])
    ls, noise = 1.0 / np.sqrt(2.0), 0.1
    kern = lambda a, b: rbf_kernel(a, b, ls)  # noqa: E731
    Xs = np.array([[0.51], [4.5], [8.0]])
    mean, var = gp_posterior(X, y, Xs, kern, noise)
    print("query x:        ", Xs[:, 0])
    print("posterior mean: ", np.round(mean, 3))
    print("posterior var:  ", np.round(var, 3))
    draws = gp_prior_samples(np.linspace(0, 6, 5)[:, None], kern, 3, np.random.default_rng(0))
    print("three prior draws at five points:\n", np.round(draws, 2))
    grid = np.geomspace(0.1, 4.0, 40)
    print("length scale with the highest evidence:", round(best_length_scale(X, y, noise, grid), 3))
