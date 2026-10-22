"""Reference solution: hold flats back, memorise the rest, and measure the gap."""

from __future__ import annotations

from typing import Callable

import numpy as np

Loss = Callable[[np.ndarray, np.ndarray], np.ndarray]


# Written in the chapter "Scoring a guess: loss and empirical risk".
def squared_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (predictions - targets) ** 2


# Written in the chapter "Scoring a guess: loss and empirical risk".
def empirical_risk(predictions: np.ndarray, targets: np.ndarray, loss: Loss) -> float:
    return float(np.mean(loss(predictions, targets)))


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split paired data into a training set and a test set with a seeded shuffle.

    Args:
        x: Flat sizes, shape (n,).
        y: Rents, shape (n,).
        test_size: Fraction of flats to hold back. The test count is the
            floor of test_size * n, computed so that 0.29 * 100 gives 29.
        seed: Seed for the shuffle. The same seed gives the same split.

    Returns:
        (x_train, y_train, x_test, y_test). Every flat is in exactly one set,
        and each size stays paired with its rent.
    """
    n = len(x)
    n_test = int(test_size * n + 1e-9)
    order = np.random.default_rng(seed).permutation(n)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def memoriser_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Guess each rent by copying the rent of the nearest training size.

    Args:
        x_train: Training sizes, shape (n,). Not necessarily sorted.
        y_train: Training rents, shape (n,).
        x_new: Sizes to guess, shape (m,).

    Returns:
        Guesses, shape (m,): the rent of the training flat closest in size.
    """
    distances = np.abs(x_new[:, None] - x_train[None, :])  # (m, n)
    nearest = np.argmin(distances, axis=1)
    return y_train[nearest]


def generalisation_gap(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[float, float, float]:
    """Split, memorise the training flats, and compare training and test risk.

    Args:
        x: Flat sizes, shape (n,).
        y: Rents, shape (n,).
        test_size: Fraction of flats to hold back.
        seed: Seed for the split.

    Returns:
        (train_risk, test_risk, gap) as Python floats, using squared loss,
        where gap = test_risk - train_risk.
    """
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size, seed)
    train_risk = empirical_risk(memoriser_predict(x_train, y_train, x_train), y_train, squared_loss)
    test_risk = empirical_risk(memoriser_predict(x_train, y_train, x_test), y_test, squared_loss)
    return train_risk, test_risk, test_risk - train_risk


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sizes = rng.uniform(0.0, 1.0, size=20)
    rents = np.sin(2.0 * np.pi * sizes) + 0.3 * rng.normal(size=20)

    x_train, y_train, x_test, y_test = train_test_split(sizes, rents, test_size=0.25, seed=0)
    print("training flats:", len(x_train), "test flats:", len(x_test))

    train_risk, test_risk, gap = generalisation_gap(sizes, rents, test_size=0.25, seed=0)
    print(f"training risk {train_risk:.3f}, test risk {test_risk:.3f}, gap {gap:.3f}")
