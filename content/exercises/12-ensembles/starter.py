"""Bagging, out-of-bag error, and gradient boosting.

`_fit_stump` and `_predict_stump` below are a complete, working depth-1
regression tree (a stump): the single axis-aligned split that minimises
weighted squared error, with each side predicting its own weighted mean. They
are the weak learner every function in this exercise builds on. You do not
need to modify them.

Fill in the five functions that use them.
"""

import numpy as np


def _fit_stump(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> dict:
    """Fit a depth-1 regression tree minimising (weighted) squared error.

    Returns a dict with keys "feature", "threshold", "left_value",
    "right_value". "feature" is `None` when no feature admits a useful split,
    in which case both values equal the (weighted) mean of `y`.
    """
    n, d = X.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    best = None

    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y[order]
        ws = w[order]

        cw = np.cumsum(ws)
        cwy = np.cumsum(ws * ys)
        cwy2 = np.cumsum(ws * ys**2)
        total_w, total_wy, total_wy2 = cw[-1], cwy[-1], cwy2[-1]

        valid = xs[:-1] != xs[1:]
        left_w, left_wy, left_wy2 = cw[:-1], cwy[:-1], cwy2[:-1]
        right_w = total_w - left_w
        right_wy = total_wy - left_wy
        right_wy2 = total_wy2 - left_wy2
        valid &= (left_w > 0) & (right_w > 0)
        if not np.any(valid):
            continue

        left_sse = left_wy2 - left_wy**2 / np.where(left_w > 0, left_w, 1.0)
        right_sse = right_wy2 - right_wy**2 / np.where(right_w > 0, right_w, 1.0)
        total_sse = np.where(valid, left_sse + right_sse, np.inf)

        i = int(np.argmin(total_sse))
        if not np.isfinite(total_sse[i]):
            continue
        candidate = (
            float(total_sse[i]),
            j,
            0.5 * (xs[i] + xs[i + 1]),
            float(left_wy[i] / left_w[i]),
            float(right_wy[i] / right_w[i]),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        value = float(np.average(y, weights=w))
        return {"feature": None, "threshold": None, "left_value": value, "right_value": value}

    _, feature, threshold, left_value, right_value = best
    return {"feature": feature, "threshold": threshold, "left_value": left_value, "right_value": right_value}


def _predict_stump(stump: dict, X: np.ndarray) -> np.ndarray:
    """Predict with a stump returned by `_fit_stump`."""
    if stump["feature"] is None:
        return np.full(X.shape[0], stump["left_value"])
    mask = X[:, stump["feature"]] <= stump["threshold"]
    return np.where(mask, stump["left_value"], stump["right_value"])


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one bootstrap resample of size n from {0, ..., n-1}.

    Args:
        n: Number of training points.
        rng: A numpy random Generator to draw from.

    Returns:
        Integer array of shape (n,), each entry drawn uniformly from
        {0, ..., n-1} independently, with replacement.
    """
    raise NotImplementedError("bootstrap_indices")


def bagging_variance(rho: float, sigma2: float, B: int) -> float:
    """The variance of an average of B estimators with pairwise correlation rho.

    Args:
        rho: Pairwise correlation between any two of the B estimators.
        sigma2: The variance of a single estimator.
        B: Number of estimators averaged.

    Returns:
        rho * sigma2 + (1 - rho) * sigma2 / B.
    """
    raise NotImplementedError("bagging_variance")


def fit_bagged_stumps(
    X: np.ndarray, y: np.ndarray, n_estimators: int, rng: np.random.Generator
) -> tuple[list[dict], list[np.ndarray]]:
    """Fit an ensemble of stumps, each on its own bootstrap resample.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        n_estimators: Number of stumps to fit.
        rng: A numpy random Generator, advanced once per estimator.

    Returns:
        A tuple `(stumps, in_bag_indices)`. `stumps` is a list of
        `n_estimators` dicts as returned by `_fit_stump`. `in_bag_indices` is
        a list of the same length, where `in_bag_indices[b]` is the array of
        row indices drawn for stump `b`'s bootstrap resample.
    """
    raise NotImplementedError("fit_bagged_stumps")


def out_of_bag_error(
    X: np.ndarray, y: np.ndarray, stumps: list[dict], in_bag_indices: list[np.ndarray]
) -> float:
    """Score a bagged ensemble using only out-of-bag trees for each point.

    For each training row i, average the predictions of every stump whose
    bootstrap resample did not include i, then compute the mean squared error
    of those averaged predictions against y, over the rows that were
    out-of-bag for at least one stump.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        stumps: Stumps returned by `fit_bagged_stumps`.
        in_bag_indices: The matching list of bootstrap index arrays.

    Returns:
        The out-of-bag mean squared error, as a float.
    """
    raise NotImplementedError("out_of_bag_error")


def gradient_boost(
    X: np.ndarray,
    y: np.ndarray,
    loss_grad,
    n_rounds: int,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, list[dict]]:
    """Stagewise functional gradient boosting with depth-1 stumps.

    Initialise the model to F = 0. At each round, compute the negative
    gradient `target = loss_grad(y, F)`, fit a stump to `(X, target)` with
    `_fit_stump`, then update `F += learning_rate * stump_predictions`.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        loss_grad: A callable `loss_grad(y, F) -> np.ndarray` returning the
            negative gradient of the loss with respect to the predictions F,
            elementwise, shape (n,). For squared error 0.5 * (y - F) ** 2,
            this is `y - F`.
        n_rounds: Number of boosting rounds.
        learning_rate: Shrinkage factor applied to every round's update.

    Returns:
        A tuple `(F, history)`. `F` is the final prediction, shape (n,).
        `history` is a list of length `n_rounds`; `history[m]` is a dict with
        keys "target" (the negative gradient fit at round m, shape (n,)) and
        "prediction" (the model's output after round m is applied, shape
        (n,)).
    """
    raise NotImplementedError("gradient_boost")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=n)

    print(f"bagging_variance(rho=0.3, sigma2=1.0, B=50) = {bagging_variance(0.3, 1.0, 50):.4f}")

    stumps, in_bag = fit_bagged_stumps(X, y, n_estimators=60, rng=rng)
    oob = out_of_bag_error(X, y, stumps, in_bag)
    print(f"out-of-bag MSE over {len(stumps)} bagged stumps: {oob:.4f}")

    F, history = gradient_boost(X, y, loss_grad=lambda yy, FF: yy - FF, n_rounds=20, learning_rate=0.3)
    losses = [0.5 * float(np.mean((y - h["prediction"]) ** 2)) for h in history]
    print(f"gradient boosting training loss: round 1 = {losses[0]:.4f}, round 20 = {losses[-1]:.4f}")
