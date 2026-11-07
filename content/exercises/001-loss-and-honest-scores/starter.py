"""Score a rule, pick its slope, and catch a memoriser.

You write six functions: `squared_loss`, `empirical_risk`, `best_slope`,
`train_test_split`, `memoriser_predict`, and `generalisation_gap`. The two
other losses are already written, so you can compare what each one cares
about.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Loss = Callable[[np.ndarray, np.ndarray], np.ndarray]


def absolute_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Given: the size of each miss, ignoring its sign. Shape (n,)."""
    return np.abs(predictions - targets)


def zero_one_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Given: 0.0 where a guess matches its target, 1.0 where it does not. Shape (n,)."""
    return (predictions != targets).astype(float)


def squared_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Score each guess by its squared miss.

    Args:
        predictions: Guesses, shape (n,).
        targets: Real values, shape (n,).

    Returns:
        One score per example, shape (n,): (prediction - target) ** 2.
    """
    raise NotImplementedError("squared_loss")


def empirical_risk(predictions: np.ndarray, targets: np.ndarray, loss: Loss) -> float:
    """Average a per-example loss over all examples.

    Args:
        predictions: Guesses, shape (n,).
        targets: Real values, shape (n,).
        loss: A function that takes (predictions, targets) and returns one
            score per example, shape (n,).

    Returns:
        The mean score as a Python float.
    """
    raise NotImplementedError("empirical_risk")


def best_slope(sizes: np.ndarray, rents: np.ndarray, slopes: np.ndarray, loss: Loss) -> float:
    """Pick the slope whose rule `rent = slope * size` has the lowest empirical risk.

    Args:
        sizes: Flat sizes, shape (n,).
        rents: Real weekly rents, shape (n,).
        slopes: Candidate slopes, shape (k,).
        loss: A per-example loss, as in `empirical_risk`.

    Returns:
        The candidate from `slopes` with the lowest risk, as a Python float.
    """
    raise NotImplementedError("best_slope")


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
    raise NotImplementedError("train_test_split")


def memoriser_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Guess each rent by copying the rent of the nearest training size.

    Args:
        x_train: Training sizes, shape (n,). Not necessarily sorted.
        y_train: Training rents, shape (n,).
        x_new: Sizes to guess, shape (m,).

    Returns:
        Guesses, shape (m,): the rent of the training flat closest in size.
    """
    raise NotImplementedError("memoriser_predict")


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
    raise NotImplementedError("generalisation_gap")


if __name__ == "__main__":
    sizes = np.array([20.0, 30.0, 40.0, 50.0])
    rents = np.array([200.0, 300.0, 400.0, 650.0])
    guesses = 11.0 * sizes  # the rule "eleven times the size"
    print("squared losses:", squared_loss(guesses, rents))
    print("empirical risk:", empirical_risk(guesses, rents, squared_loss))

    slopes = np.arange(6, 15, dtype=float)
    for name, loss in (("squared", squared_loss), ("absolute", absolute_loss)):
        print(f"best slope under {name} loss:", best_slope(sizes, rents, slopes, loss))

    rng = np.random.default_rng(0)
    flat_sizes = rng.uniform(0.0, 1.0, size=20)
    flat_rents = np.sin(2.0 * np.pi * flat_sizes) + 0.3 * rng.normal(size=20)
    x_train, y_train, x_test, y_test = train_test_split(flat_sizes, flat_rents, test_size=0.25, seed=0)
    print("training flats:", len(x_train), "test flats:", len(x_test))

    train_risk, test_risk, gap = generalisation_gap(flat_sizes, flat_rents, test_size=0.25, seed=0)
    print(f"training risk {train_risk:.3f}, test risk {test_risk:.3f}, gap {gap:.3f}")
