"""Reference solution: score a guess, average the scores, and pick the best slope."""

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
    return (predictions - targets) ** 2


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
    return float(np.mean(loss(predictions, targets)))


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
    risks = [empirical_risk(slope * sizes, rents, loss) for slope in slopes]
    return float(slopes[int(np.argmin(risks))])


if __name__ == "__main__":
    sizes = np.array([20.0, 30.0, 40.0, 50.0])
    rents = np.array([200.0, 300.0, 400.0, 650.0])
    guesses = 11.0 * sizes  # the rule "eleven times the size"
    print("squared losses:", squared_loss(guesses, rents))
    print("empirical risk:", empirical_risk(guesses, rents, squared_loss))

    slopes = np.arange(6, 15, dtype=float)
    for name, loss in (("squared", squared_loss), ("absolute", absolute_loss)):
        print(f"best slope under {name} loss:", best_slope(sizes, rents, slopes, loss))
