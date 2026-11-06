"""Fit probabilistic PCA by guess and refit.

The model says each centred point is `W z` plus Gaussian fuzz of variance
sigma2, where the hidden coordinate z is never seen. EM alternates an E-step
(guess the hidden coordinates) with an M-step (refit W and sigma2 from the
guesses). You write the E-step, the M-step, and the log-likelihood. The loop
`ppca_em` that joins them is given.
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
    raise NotImplementedError("ppca_e_step")


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
    raise NotImplementedError("ppca_m_step")


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
    raise NotImplementedError("ppca_loglik")


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


if __name__ == "__main__":
    Xc = np.array([[2.0, 0.0], [-2.0, 0.0]])
    W = np.array([[1.0], [0.0]])
    print("log-likelihood before:", ppca_loglik(Xc, W, 1.0))
    Ez, Ezz_sum = ppca_e_step(Xc, W, 1.0)
    print("Ez:", Ez.ravel(), "Ezz:", Ezz_sum.ravel())
    W_new, sigma2_new = ppca_m_step(Xc, Ez, Ezz_sum)
    print("W_new:", W_new.ravel(), "sigma2_new:", sigma2_new)
    print("log-likelihood after:", ppca_loglik(Xc, W_new, sigma2_new))

    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 2
    true_components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.array([3.0, 1.5])
    X = latent @ true_components.T + 0.2 * rng.normal(size=(n, d))
    W, sigma2, history = ppca_em(X, k, iters=300)
    print(f"sigma2 = {sigma2:.5f}, log-likelihood {history[0]:.1f} -> {history[-1]:.1f}")
    components, _, _ = pca_svd(X, k)
    Q, _ = np.linalg.qr(W)
    print("distance between planes:", np.linalg.norm(Q @ Q.T - components @ components.T))
