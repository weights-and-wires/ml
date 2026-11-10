"""Reference solution: power iteration and probabilistic PCA.

Multiply a symmetric matrix by a vector and rescale, again and again, and the
vector turns toward the eigenvector with the largest eigenvalue. The Rayleigh
quotient reads the eigenvalue, and subtracting the found part (deflation)
exposes the next eigenvector. Probabilistic PCA then adds Gaussian fuzz to the
sheet. Each centred point is `W z` plus fuzz of variance sigma2, where the
hidden coordinate z is never seen, and EM alternates an E-step (guess the
hidden coordinates) with an M-step (refit W and sigma2 from the guesses).
Factor analysis then gives each column its own noise variance, so sigma2 * I
becomes a diagonal `psi`. You write nine functions, in this order:

1. `rayleigh_quotient`, 2. `power_iteration`, 3. `deflate`,
4. `ppca_e_step`, 5. `ppca_m_step`, 6. `ppca_loglik`,
7. `fa_e_step`, 8. `fa_m_step`, 9. `fa_loglik`.

The loops `ppca_em` and `fa_em` that join each trio are given.
"""

from __future__ import annotations

import numpy as np


# Written in the chapter "Principal components: compress and know what you lost".
def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    return X - mean, mean


# Written in the chapter "Principal components: compress and know what you lost".
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


def ppca_e_step(Xc: np.ndarray, W: np.ndarray, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
    """Guess the hidden coordinates of every point.

    Args:
        Xc: Centred data, shape (n, d).
        W: Loading matrix, shape (d, k).
        sigma2: Noise variance, a positive float.

    Returns:
        A tuple `(Ez, Ezz_sum)`. With `M = W.T @ W + sigma2 * I` of shape
        (k, k), `Ez = Xc @ W @ inv(M)` has shape (n, k), and
        `Ezz_sum = n * sigma2 * inv(M) + Ez.T @ Ez` has shape (k, k).
    """
    n, k = Xc.shape[0], W.shape[1]
    M_inv = np.linalg.inv(W.T @ W + sigma2 * np.eye(k))
    Ez = Xc @ W @ M_inv  # M is symmetric, so inv(M).T equals inv(M)
    Ezz_sum = n * sigma2 * M_inv + Ez.T @ Ez
    return Ez, Ezz_sum


def ppca_m_step(Xc: np.ndarray, Ez: np.ndarray, Ezz_sum: np.ndarray) -> tuple[np.ndarray, float]:
    """Refit W and sigma2 from the guesses.

    Args:
        Xc: Centred data, shape (n, d).
        Ez: Guessed hidden coordinates, shape (n, k).
        Ezz_sum: The summed second moments, shape (k, k).

    Returns:
        A tuple `(W_new, sigma2_new)`. `W_new = Xc.T @ Ez @ inv(Ezz_sum)` has
        shape (d, k). `sigma2_new` is
        `(sum(Xc**2) - 2 * sum(Ez * (Xc @ W_new)) + trace(Ezz_sum @ W_new.T @ W_new)) / (n * d)`,
        using `W_new` in every term, and clipped below at `1e-8`.
    """
    n, d = Xc.shape
    W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz_sum)
    squared_data = float(np.sum(Xc**2))
    cross = 2.0 * float(np.sum(Ez * (Xc @ W_new)))
    fit = float(np.trace(Ezz_sum @ (W_new.T @ W_new)))
    sigma2_new = max((squared_data - cross + fit) / (n * d), 1e-8)
    return W_new, sigma2_new


def ppca_loglik(Xc: np.ndarray, W: np.ndarray, sigma2: float) -> float:
    """Log-likelihood of centred data under probabilistic PCA.

    Args:
        Xc: Centred data, shape (n, d).
        W: Loading matrix, shape (d, k).
        sigma2: Noise variance, a positive float.

    Returns:
        With `C = W @ W.T + sigma2 * I` and `S = Xc.T @ Xc / n`, the value
        `-n / 2 * (d * log(2 pi) + logdet(C) + trace(inv(C) @ S))`. Use
        `np.linalg.slogdet` and `np.linalg.solve`.
    """
    n, d = Xc.shape
    C = W @ W.T + sigma2 * np.eye(d)
    S = (Xc.T @ Xc) / n
    _, logdet = np.linalg.slogdet(C)
    trace_term = np.trace(np.linalg.solve(C, S))
    return float(-0.5 * n * (d * np.log(2.0 * np.pi) + logdet + trace_term))


def fa_e_step(Xc: np.ndarray, W: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Guess the hidden coordinates when each column has its own noise variance.

    Args:
        Xc: Centred data, shape (n, d).
        W: Loading matrix, shape (d, k).
        psi: Noise variance of each column, shape (d,), all positive.

    Returns:
        A tuple `(Ez, Ezz_sum)`. With `P = W / psi[:, None]` (each row of `W`
        divided by its noise variance) and `M = I + W.T @ P` of shape (k, k),
        `Ez = Xc @ P @ inv(M)` has shape (n, k), and
        `Ezz_sum = n * inv(M) + Ez.T @ Ez` has shape (k, k).
    """
    n, k = Xc.shape[0], W.shape[1]
    P = W / psi[:, None]
    M_inv = np.linalg.inv(np.eye(k) + W.T @ P)
    Ez = Xc @ P @ M_inv
    Ezz_sum = n * M_inv + Ez.T @ Ez
    return Ez, Ezz_sum


def fa_m_step(Xc: np.ndarray, Ez: np.ndarray, Ezz_sum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refit W and the per-column noise variances from the guesses.

    Args:
        Xc: Centred data, shape (n, d).
        Ez: Guessed hidden coordinates, shape (n, k).
        Ezz_sum: The summed second moments, shape (k, k).

    Returns:
        A tuple `(W_new, psi_new)`. `W_new = Xc.T @ Ez @ inv(Ezz_sum)` has
        shape (d, k). `psi_new[j]` is
        `(sum(Xc[:, j]**2) - W_new[j] @ (Xc.T @ Ez)[j]) / n`, which is the
        diagonal of the leftover covariance. Clip it below at `1e-6`.
    """
    n = Xc.shape[0]
    cross = Xc.T @ Ez
    W_new = cross @ np.linalg.inv(Ezz_sum)
    psi_new = (np.sum(Xc**2, axis=0) - np.sum(W_new * cross, axis=1)) / n
    return W_new, np.maximum(psi_new, 1e-6)


def fa_loglik(Xc: np.ndarray, W: np.ndarray, psi: np.ndarray) -> float:
    """Log-likelihood of centred data under factor analysis.

    Args:
        Xc: Centred data, shape (n, d).
        W: Loading matrix, shape (d, k).
        psi: Noise variance of each column, shape (d,).

    Returns:
        The same formula as `ppca_loglik`, with `C = W @ W.T + np.diag(psi)`.
    """
    n, d = Xc.shape
    C = W @ W.T + np.diag(psi)
    S = (Xc.T @ Xc) / n
    _, logdet = np.linalg.slogdet(C)
    trace_term = np.trace(np.linalg.solve(C, S))
    return float(-0.5 * n * (d * np.log(2.0 * np.pi) + logdet + trace_term))


# Given: this loop calls your three functions. Read it once they work.
def ppca_em(X: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, float, list[float]]:
    """Fit probabilistic PCA by EM.

    Args:
        X: Data matrix, shape (n, d).
        k: Hidden dimension.
        iters: Number of EM rounds.

    Returns:
        A tuple `(W, sigma2, history)`: the fitted loading matrix (d, k), the
        noise variance, and the log-likelihood after each round.
    """
    Xc, _ = center(X)
    d = Xc.shape[1]
    W = np.random.default_rng(0).normal(scale=0.1, size=(d, k))
    sigma2 = 1.0
    history = []
    for _ in range(iters):
        Ez, Ezz_sum = ppca_e_step(Xc, W, sigma2)
        W, sigma2 = ppca_m_step(Xc, Ez, Ezz_sum)
        history.append(float(ppca_loglik(Xc, W, sigma2)))
    return W, sigma2, history


# Given: the same loop with a diagonal noise covariance. Read it once the three work.
def fa_em(X: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit factor analysis by EM.

    Args:
        X: Data matrix, shape (n, d).
        k: Hidden dimension.
        iters: Number of EM rounds.

    Returns:
        A tuple `(W, psi, history)`: the loading matrix (d, k), the per-column
        noise variances (d,), and the log-likelihood after each round.
    """
    Xc, _ = center(X)
    d = Xc.shape[1]
    W = np.random.default_rng(0).normal(scale=0.1, size=(d, k))
    psi = np.ones(d)
    history = []
    for _ in range(iters):
        Ez, Ezz_sum = fa_e_step(Xc, W, psi)
        W, psi = fa_m_step(Xc, Ez, Ezz_sum)
        history.append(float(fa_loglik(Xc, W, psi)))
    return W, psi, history


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

    Xc = np.array([[2.0, 0.0], [-2.0, 0.0]])
    W = np.array([[1.0], [0.0]])
    print("log-likelihood before:", ppca_loglik(Xc, W, 1.0))
    Ez, Ezz_sum = ppca_e_step(Xc, W, 1.0)
    print("Ez:", Ez.ravel(), "Ezz:", Ezz_sum.ravel())
    W_new, sigma2_new = ppca_m_step(Xc, Ez, Ezz_sum)
    print("W_new:", W_new.ravel(), "sigma2_new:", sigma2_new)
    print("log-likelihood after:", ppca_loglik(Xc, W_new, sigma2_new))

    W, sigma2, history = ppca_em(X, k, iters=300)
    print(f"sigma2 = {sigma2:.5f}, log-likelihood {history[0]:.1f} -> {history[-1]:.1f}")
    Q, _ = np.linalg.qr(W)
    print("distance between planes:", np.linalg.norm(Q @ Q.T - components @ components.T))
