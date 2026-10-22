"""Reference solution: split the expected squared error into bias, variance, and noise."""

from __future__ import annotations

from typing import Callable

import numpy as np


# Written in the chapter "Fitting a curve: polynomials, least squares, and the degree dial".
def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    return np.asarray(x, dtype=float)[:, None] ** np.arange(degree + 1)[None, :]


# Written in the chapter "Fitting a curve: polynomials, least squares, and the degree dial".
def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(polynomial_features(x, degree), y, rcond=None)[0]


# Written in the chapter "Fitting a curve: polynomials, least squares, and the degree dial".
def predict_polynomial(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return polynomial_features(x, len(w) - 1) @ w


def dart_decomposition(
    hits: np.ndarray, bullseye: float, noise_std: float
) -> tuple[float, float, float]:
    """Split the expected squared miss for repeated guesses at one flat.

    Args:
        hits: Guesses of several fits for the same flat, shape (k,).
        bullseye: The true rent f(x) of that flat, with no noise.
        noise_std: Standard deviation of the label noise.

    Returns:
        (bias_sq, variance, noise) as Python floats: the squared gap between
        the average guess and the bullseye, the spread of the guesses around
        their own average (divide by k, not k - 1), and noise_std ** 2.
    """
    hits = np.asarray(hits, dtype=float)
    bias_sq = (hits.mean() - bullseye) ** 2
    variance = np.var(hits)
    return float(bias_sq), float(variance), float(noise_std**2)


def sample_training_set(
    true_function: Callable[[np.ndarray], np.ndarray],
    noise_std: float,
    n_train: int,
    x_domain: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one training set from the market.

    Args:
        true_function: The noise-free rent f, vectorised over sizes.
        noise_std: Standard deviation of the Gaussian label noise.
        n_train: Number of flats to draw.
        x_domain: (low, high) bounds for the sizes, drawn uniformly.
        rng: The random generator to draw from. Use it, and do not make a new one.

    Returns:
        (x, y), each shape (n_train,): sizes first drawn from rng.uniform, then
        y = true_function(x) + noise_std * rng.normal(size=n_train).
    """
    x = rng.uniform(x_domain[0], x_domain[1], size=n_train)
    y = true_function(x) + noise_std * rng.normal(size=n_train)
    return x, y


def bias_variance_decomposition(
    true_function: Callable[[np.ndarray], np.ndarray],
    noise_std: float,
    x_query: np.ndarray,
    degree: int,
    n_train: int,
    n_trials: int,
    x_domain: tuple[float, float] = (0.0, 1.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate squared bias, variance, and noise at each query size by simulation.

    Draws `n_trials` training sets with one generator, np.random.default_rng(seed),
    fits a degree-`degree` polynomial to each, and records its guesses at x_query.

    Args:
        true_function: The noise-free rent f.
        noise_std: Standard deviation of the label noise.
        x_query: Sizes at which to measure, shape (q,).
        degree: Polynomial degree of every fit.
        n_train: Flats per training set.
        n_trials: Number of training sets.
        x_domain: (low, high) for the training sizes.
        seed: Seed for the single generator.

    Returns:
        (bias_sq, variance, noise), each shape (q,).
    """
    rng = np.random.default_rng(seed)
    x_query = np.asarray(x_query, dtype=float)
    predictions = np.empty((n_trials, len(x_query)))
    for trial in range(n_trials):
        x_train, y_train = sample_training_set(true_function, noise_std, n_train, x_domain, rng)
        w = fit_polynomial(x_train, y_train, degree)
        predictions[trial] = predict_polynomial(w, x_query)
    mean_prediction = predictions.mean(axis=0)
    bias_sq = (mean_prediction - true_function(x_query)) ** 2
    variance = predictions.var(axis=0)
    noise = np.full(len(x_query), noise_std**2)
    return bias_sq, variance, noise


if __name__ == "__main__":
    hits = np.array([0.8, 1.2, 1.0, 1.4])
    bias_sq, variance, noise = dart_decomposition(hits, bullseye=1.0, noise_std=0.3)
    print(f"darts: bias^2 {bias_sq:.3f}, variance {variance:.3f}, noise {noise:.3f}")

    wave = lambda size: np.sin(2.0 * np.pi * size)
    x_query = np.array([0.25])
    for degree in (1, 3, 5):
        b, v, n = bias_variance_decomposition(wave, 0.3, x_query, degree, n_train=15, n_trials=2000, seed=0)
        print(f"degree {degree}: bias^2 {b[0]:.3f}, variance {v[0]:.3f}, noise {n[0]:.3f}, total {b[0] + v[0] + n[0]:.3f}")
