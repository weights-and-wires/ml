"""Risk, empirical risk, and the bias-variance decomposition.

Fill in the five functions below. The first four are the tools of empirical
risk minimisation: a way to score predictions, a way to hold out data
honestly, and a small hypothesis class (polynomials) to fit it with. The
fifth runs a simulation that splits the expected squared error into
squared bias, variance, and noise, as the lesson's "Bias, variance, and
noise" section describes, so you can check the split with real numbers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def empirical_risk(
    predictions: np.ndarray, targets: np.ndarray, loss: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> float:
    """Average a per-sample loss over a batch of predictions and targets.

    Args:
        predictions: Model outputs, shape (n,).
        targets: True labels, shape (n,).
        loss: A vectorised function taking (predictions, targets), both shape
            (n,), and returning a per-sample loss, shape (n,).

    Returns:
        The empirical risk, `mean(loss(predictions, targets))`, as a float.
    """
    raise NotImplementedError("empirical_risk")


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition (x, y) into a training set and a held-out set.

    Args:
        x: Inputs, shape (n, ...).
        y: Targets, shape (n,).
        test_size: Fraction of the n samples to place in the held-out set,
            rounded down to the nearest integer count.
        seed: Seed for the random permutation, so the split is reproducible.

    Returns:
        A tuple (x_train, y_train, x_test, y_test). Every original index
        appears in exactly one of the two splits.
    """
    raise NotImplementedError("train_test_split")


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Build the polynomial design matrix [1, x, x^2, ..., x^degree].

    Args:
        x: Inputs, shape (n,).
        degree: The highest power to include.

    Returns:
        A matrix of shape (n, degree + 1) whose column j is x ** j.
    """
    raise NotImplementedError("polynomial_features")


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Fit a degree-`degree` polynomial to (x, y) by least squares.

    Args:
        x: Inputs, shape (n,).
        y: Targets, shape (n,).
        degree: The polynomial degree to fit.

    Returns:
        Coefficients w, shape (degree + 1,), such that
        `polynomial_features(x, degree) @ w` approximates y in the least
        squares sense.
    """
    raise NotImplementedError("fit_polynomial")


def bias_variance_decomposition(
    true_function: Callable[[np.ndarray], np.ndarray],
    noise_std: float,
    x_query: np.ndarray,
    degree: int,
    n_train: int,
    n_trials: int,
    x_domain: tuple[float, float] = (-1.0, 1.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate squared bias, variance, and noise at each point in x_query.

    Runs a Monte Carlo experiment: for each of `n_trials` independent
    training sets of size `n_train`, drawn with inputs uniform on
    `x_domain` and labels `true_function(x) + noise`, fit a degree-`degree`
    polynomial and record its predictions at every point in `x_query`. The
    squared bias is the squared gap between the mean prediction and
    `true_function`; the variance is the spread of predictions around their
    own mean; the noise is the irreducible label noise.

    Args:
        true_function: The regression function f, vectorised over its input.
        noise_std: Standard deviation of the mean-zero Gaussian label noise.
        x_query: Points at which to evaluate the decomposition, shape (q,).
        degree: Degree of the polynomial fit on each resampled training set.
        n_train: Number of training points drawn per trial.
        n_trials: Number of independent resampled training sets.
        x_domain: (low, high) bounds for sampling training inputs uniformly.
        seed: Seed for the random number generator driving every draw.

    Returns:
        A tuple (bias_sq, variance, noise), each shape (q,), satisfying
        bias_sq + variance + noise approx equal to the expected squared error
        of the fitted polynomial against a fresh label at each query point.
    """
    raise NotImplementedError("bias_variance_decomposition")


if __name__ == "__main__":
    def true_function(x: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * x)

    x_query = np.linspace(-0.9, 0.9, 9)

    for degree in (1, 3, 9):
        bias_sq, variance, noise = bias_variance_decomposition(
            true_function,
            noise_std=0.2,
            x_query=x_query,
            degree=degree,
            n_train=15,
            n_trials=500,
            seed=0,
        )
        total = bias_sq.mean() + variance.mean() + noise.mean()
        print(
            f"degree {degree:>2}: bias^2 {bias_sq.mean():.4f}  "
            f"variance {variance.mean():.4f}  noise {noise.mean():.4f}  "
            f"total {total:.4f}"
        )

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=200)
    y = true_function(x) + 0.2 * rng.normal(size=200)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3, seed=1)
    w = fit_polynomial(x_train, y_train, degree=9)
    squared_error = lambda pred, target: (pred - target) ** 2
    train_risk = empirical_risk(polynomial_features(x_train, 9) @ w, y_train, squared_error)
    test_risk = empirical_risk(polynomial_features(x_test, 9) @ w, y_test, squared_error)
    print(f"train risk {train_risk:.4f}, test risk {test_risk:.4f}")
