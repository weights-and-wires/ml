"""Reference solution: find the top eigenvector with one repeated product.

Multiply a vector by a symmetric matrix and rescale it, again and again, and it
turns toward the eigenvector with the largest eigenvalue. The Rayleigh quotient
reads the eigenvalue, and subtracting the found part (deflation) exposes the
next eigenvector. You build all three pieces.
"""

from __future__ import annotations

import numpy as np


# Written in the chapter "Principal directions: where the data spreads".
def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    return X - mean, mean


# Written in the chapter "Principal directions: where the data spreads".
def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc, _ = center(X)
    n = X.shape[0]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].T
    scores = Xc @ components
    return components, scores, s[:k] ** 2 / (n - 1)


def rayleigh_quotient(A: np.ndarray, v: np.ndarray) -> float:
    """The Rayleigh quotient of a symmetric matrix at a vector.

    Args:
        A: A symmetric matrix, shape (d, d).
        v: A nonzero vector, shape (d,). It need not have length 1.

    Returns:
        `v @ A @ v / (v @ v)` as a float. At an eigenvector it equals the
        eigenvalue.
    """
    return float(v @ A @ v / (v @ v))


def power_iteration(A: np.ndarray, iters: int) -> tuple[np.ndarray, float]:
    """Find the eigenvector of largest magnitude by repeated multiplication.

    Start from the fixed vector `np.ones(d) / np.sqrt(d)`. At each of `iters`
    steps, multiply by `A` and divide by the length of the result.

    Args:
        A: A symmetric matrix, shape (d, d).
        iters: Number of steps to run.

    Returns:
        A tuple `(v, eigenvalue)`. `v` is a unit vector, shape (d,). `eigenvalue`
        is `rayleigh_quotient(A, v)`.
    """
    d = A.shape[0]
    v = np.ones(d) / np.sqrt(d)
    for _ in range(iters):
        v = A @ v
        v = v / np.linalg.norm(v)
    return v, rayleigh_quotient(A, v)


def deflate(A: np.ndarray, v: np.ndarray, eigenvalue: float) -> np.ndarray:
    """Remove the part of A along a unit eigenvector.

    Args:
        A: A symmetric matrix, shape (d, d).
        v: A unit eigenvector of A, shape (d,).
        eigenvalue: The eigenvalue that goes with `v`.

    Returns:
        `A - eigenvalue * np.outer(v, v)`, shape (d, d). Its top eigenvalue is
        the second eigenvalue of `A`.
    """
    return A - eigenvalue * np.outer(v, v)


if __name__ == "__main__":
    A = np.diag([2.0, 1.0])
    for steps in (1, 2, 3):
        v, eigenvalue = power_iteration(A, steps)
        print(f"after {steps} step(s): v = {np.round(v, 4)}, eigenvalue estimate {eigenvalue:.4f}")

    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 2
    true_components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.array([3.0, 1.5])
    X = latent @ true_components.T + 0.2 * rng.normal(size=(n, d))
    C = np.cov(X, rowvar=False)

    v1, l1 = power_iteration(C, 200)
    v2, l2 = power_iteration(deflate(C, v1, l1), 200)
    components, _, variances = pca_svd(X, 2)
    print(f"power iteration: {l1:.4f}, {l2:.4f}")
    print(f"SVD variances:   {variances[0]:.4f}, {variances[1]:.4f}")
    print(f"|v1 . pc1| = {abs(v1 @ components[:, 0]):.6f}, |v2 . pc2| = {abs(v2 @ components[:, 1]):.6f}")
