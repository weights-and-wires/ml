"""Fit polynomials by least squares and sweep the degree dial.

You write three functions: `polynomial_features`, `fit_polynomial`, and
`sweep_degrees`. The functions from earlier chapters and `predict_polynomial`
are already here.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Loss = Callable[[np.ndarray, np.ndarray], np.ndarray]


# Written in the chapter "Scoring a guess: loss and empirical risk".
def squared_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (predictions - targets) ** 2


# Written in the chapter "Scoring a guess: loss and empirical risk".
def empirical_risk(predictions: np.ndarray, targets: np.ndarray, loss: Loss) -> float:
    return float(np.mean(loss(predictions, targets)))


# Written in the chapter "Honest scores: training error, test error, and holding data back".
def train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    n_test = int(test_size * n + 1e-9)
    order = np.random.default_rng(seed).permutation(n)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Build the design matrix with one column per power of x.

    Args:
        x: Flat sizes, shape (n,).
        degree: The highest power.

    Returns:
        Shape (n, degree + 1), where column j is x ** j.
    """
    raise NotImplementedError("polynomial_features")


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Fit the weights of a degree-`degree` polynomial by least squares.

    Args:
        x: Flat sizes, shape (n,).
        y: Rents, shape (n,).
        degree: The polynomial degree.

    Returns:
        Weights, shape (degree + 1,), that minimise the total squared miss
        between polynomial_features(x, degree) @ w and y.
    """
    raise NotImplementedError("fit_polynomial")


def predict_polynomial(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Given: evaluate the polynomial with weights w at sizes x. Shape (n,).

    It uses your `polynomial_features`, so it works once that function does.
    """
    return polynomial_features(x, len(w) - 1) @ w


def sweep_degrees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degrees: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit each degree on the training flats and score it on both sets.

    Args:
        x_train, y_train: Training flats, shape (n_train,).
        x_test, y_test: Held-out flats, shape (n_test,).
        degrees: The degrees to try.

    Returns:
        (train_risks, test_risks), each shape (len(degrees),), using squared
        loss. Every fit uses the training flats only.
    """
    raise NotImplementedError("sweep_degrees")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wave = lambda size: np.sin(2.0 * np.pi * size)
    x_train = rng.uniform(0.0, 1.0, size=15)
    y_train = wave(x_train) + 0.3 * rng.normal(size=15)
    x_test = rng.uniform(0.0, 1.0, size=200)
    y_test = wave(x_test) + 0.3 * rng.normal(size=200)

    degrees = list(range(1, 13))
    train_risks, test_risks = sweep_degrees(x_train, y_train, x_test, y_test, degrees)
    for degree, train, test in zip(degrees, train_risks, test_risks):
        print(f"degree {degree:>2}: training risk {train:.4f}, held-out risk {test:.3f}")
    print("best degree by held-out risk:", degrees[int(np.argmin(test_risks))])
