"""Autoencoders and variational autoencoders.

gaussian_kl is provided. The other functions are yours.
"""

from __future__ import annotations

import numpy as np


def gaussian_kl(mu: np.ndarray, logvar: np.ndarray) -> float:
    """KL of N(mu, diag(exp(logvar))) against the standard normal.

    The sum runs over every element. A matrix of codes returns the sum of
    the per-code KL values, not the mean.

    Args:
        mu: Posterior means, any shape.
        logvar: Log-variances, the same shape as mu.

    Returns:
        0.5 * sum(mu^2 + exp(logvar) - 1 - logvar), as a Python float.
    """
    mu_a = np.asarray(mu, dtype=float)
    logvar_a = np.asarray(logvar, dtype=float)
    if mu_a.shape != logvar_a.shape:
        raise ValueError(f"mu shape {mu_a.shape} does not match logvar shape {logvar_a.shape}")
    return float(0.5 * np.sum(mu_a**2 + np.exp(logvar_a) - 1.0 - logvar_a))


def gaussian_kl_grad(mu: np.ndarray, logvar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of gaussian_kl with respect to mu and logvar.

    Args:
        mu: Posterior means, any shape.
        logvar: Log-variances, the same shape as mu.

    Returns:
        A pair of arrays of the same shape as mu. The first is the gradient
        with respect to mu, and the second is the gradient with respect to
        logvar.
    """
    raise NotImplementedError("write gaussian_kl_grad")


def reparameterise(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Draw a diagonal-Gaussian sample with the noise held fixed.

    Args:
        mu: Means, any shape.
        logvar: Log-variances, the same shape as mu.
        eps: Standard-normal noise, the same shape as mu.

    Returns:
        An array the same shape as mu.
    """
    raise NotImplementedError("write reparameterise")


def reparameterise_grad(
    mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise derivatives of reparameterise at a fixed eps.

    Args:
        mu: Means, any shape.
        logvar: Log-variances, the same shape as mu.
        eps: The noise held fixed, the same shape as mu.

    Returns:
        A pair of arrays of the same shape as mu. The first is the derivative
        with respect to mu, and the second is the derivative with respect to
        logvar.
    """
    raise NotImplementedError("write reparameterise_grad")


def elbo(
    x: np.ndarray, mu_q: np.ndarray, logvar_q: np.ndarray, reconstruction: np.ndarray
) -> float:
    """Evidence lower bound, reconstruction term minus the KL.

    Args:
        x: The observation the reconstruction term belongs to. A scalar
            reconstruction is the expected log likelihood of this observation.
            An array must have the same shape as x.
        mu_q: Posterior means.
        logvar_q: Posterior log-variances, the same shape as mu_q.
        reconstruction: Expected log p(x | z) under q, a float or an array
            the same shape as x.

    Returns:
        One Python float.
    """
    raise NotImplementedError("write elbo")


def linear_autoencoder_fit(X: np.ndarray, rank: int) -> np.ndarray:
    """Decoder whose columns span the top principal components of X.

    Args:
        X: Data matrix, shape (n, d), one row per point.
        rank: How many components to keep.

    Returns:
        Decoder, shape (d, rank).
    """
    raise NotImplementedError("write linear_autoencoder_fit")


def train_vae(
    X: np.ndarray, steps: int, seed: int, beta: float = 1.0
) -> tuple[np.ndarray, float, float]:
    """Train a linear-Gaussian VAE and draw decoder samples.

    Args:
        X: Data matrix, shape (n, 2).
        steps: Gradient steps.
        seed: Seed for the initial weights and for the samples.
        beta: Weight on the KL. The default is 1.

    Returns:
        samples: Array of shape (at least 200, 2).
        reconstruction: Mean expected negative log likelihood, a cost.
        kl: Mean KL of q against the standard normal.
    """
    raise NotImplementedError("write train_vae")


if __name__ == "__main__":
    origin = gaussian_kl(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    print("KL at mu = 0, logvar = 0:", origin)
    sample = reparameterise(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    print("sample at eps = 1:", float(sample))
