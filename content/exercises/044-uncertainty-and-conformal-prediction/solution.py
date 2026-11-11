"""Reference solution: ensembles, MC dropout, and split conformal prediction."""

from __future__ import annotations

import math

import numpy as np


def ensemble_predict(models, X: np.ndarray):
    """Return the mean and variance of the models' predictions.

    Args:
        models: A list of callables. Each maps an array of shape `(n,)` to
            predictions of shape `(n,)`.
        X: Inputs of shape `(n,)`.

    Returns:
        A pair `(mean, var)`, each of shape `(n,)`. The variance divides by the
        number of models.
    """
    preds = np.stack([np.asarray(m(X), dtype=float) for m in models])
    return preds.mean(axis=0), preds.var(axis=0)


def mc_dropout_predict(forward, X: np.ndarray, p: float, n: int, rng: np.random.Generator):
    """Average `n` stochastic forward passes with dropout switched on.

    Args:
        forward: A callable `forward(X, p, rng)` that runs one pass with a fresh
            dropout mask and returns outputs of shape `(len(X),)`.
        X: Inputs.
        p: Probability of dropping a unit, passed to `forward`.
        n: Number of passes.
        rng: A `numpy.random.Generator`, passed to every call of `forward`.

    Returns:
        A pair `(mean, var)`, each of shape `(len(X),)`, taken over the passes.
    """
    passes = np.stack([np.asarray(forward(X, p, rng), dtype=float) for _ in range(n)])
    return passes.mean(axis=0), passes.var(axis=0)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the k-th smallest score with k = ceil((n + 1)(1 - alpha)).

    Args:
        scores: Calibration scores, shape `(n,)`.
        alpha: The miss rate, in `(0, 1)`.

    Returns:
        The quantile, or `np.inf` when k exceeds n.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def conformal_interval(pred_cal: np.ndarray, y_cal: np.ndarray, pred_test: np.ndarray, alpha: float):
    """Build split conformal intervals for regression.

    Args:
        pred_cal: Model predictions on the calibration inputs, shape `(n,)`.
        y_cal: Calibration labels, shape `(n,)`.
        pred_test: Model predictions on the test inputs, shape `(m,)`.
        alpha: The miss rate.

    Returns:
        A pair `(lo, hi)`, each of shape `(m,)`.
    """
    q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)), alpha)
    pred_test = np.asarray(pred_test, dtype=float)
    return pred_test - q, pred_test + q


def conformal_sets(probs_cal: np.ndarray, y_cal: np.ndarray, probs_test: np.ndarray, alpha: float) -> np.ndarray:
    """Build conformal prediction sets for classification.

    Args:
        probs_cal: Class probabilities on the calibration set, shape `(n, K)`.
        y_cal: Integer calibration labels, shape `(n,)`.
        probs_test: Class probabilities on the test set, shape `(m, K)`.
        alpha: The miss rate.

    Returns:
        A boolean array of shape `(m, K)`, true where the class is in the set.
    """
    probs_cal = np.asarray(probs_cal, dtype=float)
    scores = 1.0 - probs_cal[np.arange(len(y_cal)), np.asarray(y_cal)]
    q = conformal_quantile(scores, alpha)
    return (1.0 - np.asarray(probs_test, dtype=float)) <= q


if __name__ == "__main__":
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.1, 1.5])
    print("quantile at alpha = 0.2:", conformal_quantile(scores, 0.2))
    lo, hi = conformal_interval(np.zeros(9), scores, np.array([3.0]), 0.2)
    print("interval around 3.0:", lo, hi)
