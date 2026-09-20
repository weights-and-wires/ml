"""Reference solution for risk, empirical risk, and the bias-variance split."""

from __future__ import annotations

from typing import Callable

import numpy as np


def empirical_risk(
    predictions: np.ndarray, targets: np.ndarray, loss: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> float:
    return float(np.mean(loss(predictions, targets)))


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    n_test = int(np.floor(test_size * n))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    powers = np.arange(degree + 1)
    return x[:, None] ** powers[None, :]


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    phi = polynomial_features(x, degree)
    w, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return w


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
    rng = np.random.default_rng(seed)
    x_query = np.asarray(x_query, dtype=float)
    phi_query = polynomial_features(x_query, degree)

    predictions = np.empty((n_trials, x_query.shape[0]))
    for trial in range(n_trials):
        x_train = rng.uniform(x_domain[0], x_domain[1], size=n_train)
        y_train = true_function(x_train) + noise_std * rng.normal(size=n_train)
        w = fit_polynomial(x_train, y_train, degree)
        predictions[trial] = phi_query @ w

    mean_prediction = predictions.mean(axis=0)
    bias_sq = (mean_prediction - true_function(x_query)) ** 2
    variance = predictions.var(axis=0)
    noise = np.full_like(x_query, noise_std ** 2)
    return bias_sq, variance, noise


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
