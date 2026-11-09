"""Likelihood, MAP, and the predictive variance of Bayesian linear regression.

Fill in the six functions below, in order. The first scores Gaussian noise.
The next two add a Gaussian prior and its slope. The last three return the
ridge weights, the posterior bell, and the spread at a house you have not fitted.
"""

from __future__ import annotations

import numpy as np


def neg_log_likelihood_gaussian(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, sigma2: float
) -> float:
    """Negative log likelihood of i.i.d. Gaussian noise with variance sigma2.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        sigma2: Noise variance, sigma2 > 0.

    Returns:
        0.5 * n * log(2 * pi * sigma2) + 0.5 * ||X @ w - y||^2 / sigma2,
        as a Python float.
    """
    raise NotImplementedError("neg_log_likelihood_gaussian")


def map_objective(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, sigma2: float, tau2: float
) -> float:
    """Negative log of likelihood times a N(0, tau2 I) prior.

    The marginal probability of y does not depend on w, so the weights that
    minimise this score are the MAP weights.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        sigma2: Noise variance, sigma2 > 0.
        tau2: Prior variance of each weight, tau2 > 0.

    Returns:
        neg_log_likelihood_gaussian(X, y, w, sigma2)
        + 0.5 * d * log(2 * pi * tau2) + 0.5 * ||w||^2 / tau2,
        as a Python float.
    """
    raise NotImplementedError("map_objective")


def map_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, sigma2: float, tau2: float
) -> np.ndarray:
    """Gradient of map_objective with respect to the weights.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        sigma2: Noise variance, sigma2 > 0.
        tau2: Prior variance of each weight, tau2 > 0.

    Returns:
        X.T @ (X @ w - y) / sigma2 + w / tau2, shape (d,).
    """
    raise NotImplementedError("map_gradient")


def ridge_weights(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve the ridge weights for one fee.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Ridge fee, lam >= 0. For MAP, pass sigma2 / tau2.

    Returns:
        The solution of (X.T @ X + lam * I) @ w = X.T @ y, shape (d,).
    """
    raise NotImplementedError("ridge_weights")


def posterior_linreg(
    X: np.ndarray, y: np.ndarray, sigma2: float, tau2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and covariance for Gaussian noise and a N(0, tau2 I) prior.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        sigma2: Noise variance, sigma2 > 0.
        tau2: Prior variance of each weight, tau2 > 0.

    Returns:
        mean, shape (d,), the ridge solution at lam = sigma2 / tau2.
        cov, shape (d, d), equal to sigma2 * (X.T @ X + lam * I)^-1.
    """
    raise NotImplementedError("posterior_linreg")


def predictive(
    x_new: np.ndarray, mean: np.ndarray, cov: np.ndarray, sigma2: float
) -> tuple[float, float]:
    """Predictive mean and variance at one new row.

    Args:
        x_new: One row, shape (d,).
        mean: Posterior mean, shape (d,).
        cov: Posterior covariance, shape (d, d).
        sigma2: Noise variance, sigma2 > 0.

    Returns:
        pred_mean = x_new @ mean, and pred_var = sigma2 + x_new @ cov @ x_new,
        as two Python floats.
    """
    raise NotImplementedError("predictive")


if __name__ == "__main__":
    # The two houses: a studio at (size 0, price 0) and size 2 sold at price 2.
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    y = np.array([0.0, 2.0])
    sigma2, tau2 = 1.0, 1.0
    w = np.array([2.0 / 11.0, 8.0 / 11.0])
    print("negative log likelihood", neg_log_likelihood_gaussian(X, y, w, sigma2))
    print("map objective", map_objective(X, y, w, sigma2, tau2))
    print("map gradient", map_gradient(X, y, w, sigma2, tau2))
    fitted = ridge_weights(X, y, sigma2 / tau2)
    print("ridge weights", fitted)
    mean, cov = posterior_linreg(X, y, sigma2, tau2)
    print("posterior covariance trace", float(np.trace(cov)))
    for size in (0.0, 2.0 / 3.0, 6.0):
        pred_mean, pred_var = predictive(np.array([1.0, size]), mean, cov, sigma2)
        print(f"size {size:.3f}  mean {pred_mean:.4f}  variance {pred_var:.4f}")
