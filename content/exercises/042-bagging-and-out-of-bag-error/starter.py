"""Bagging, and the free test set it leaves behind.

One dataset becomes many full-size training sets when you draw rows with
replacement. You fit one stump on each draw and average the stumps. The rows a
draw misses then score the ensemble without holding any data back. You write
six functions, in this order:

1. `bootstrap_indices`: one draw of n row numbers, with replacement.
2. `fit_bagged_stumps`: one stump per draw, keeping the draws.
3. `bagged_predict`: the mean of the stumps' predictions.
4. `oob_mask`: which rows one draw left out.
5. `oob_predictions`: for each row, the mean over the stumps that missed it.
6. `out_of_bag_error`: the mean squared error of those predictions.

`_fit_stump`, `_predict_stump`, and `in_bag_fraction` are already written.
"""

import numpy as np


# --- Provided helpers: a depth-1 regression tree and the in-bag fraction. ---


def _fit_stump(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a depth-1 regression tree: the single split that minimises squared error.

    Returns a dict with keys "feature", "threshold", "left_value", and
    "right_value". A point goes left when `X[:, feature] <= threshold`, and each
    side predicts the mean of its targets. "feature" is `None` when no feature
    admits a split, and then both values equal the mean of `y`.
    """
    n, d = X.shape
    best = None
    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y[order]
        cumsum = np.cumsum(ys)
        cumsum2 = np.cumsum(ys ** 2)
        total, total2 = cumsum[-1], cumsum2[-1]

        counts = np.arange(1, n)
        left_sum, left_sum2 = cumsum[:-1], cumsum2[:-1]
        right_sum, right_sum2 = total - left_sum, total2 - left_sum2
        right_counts = n - counts
        sse = (left_sum2 - left_sum ** 2 / counts) + (right_sum2 - right_sum ** 2 / right_counts)
        sse = np.where(xs[:-1] != xs[1:], sse, np.inf)

        i = int(np.argmin(sse))
        if not np.isfinite(sse[i]):
            continue
        if best is None or sse[i] < best[0]:
            best = (
                float(sse[i]),
                j,
                float(0.5 * (xs[i] + xs[i + 1])),
                float(left_sum[i] / counts[i]),
                float(right_sum[i] / right_counts[i]),
            )

    if best is None:
        value = float(np.mean(y))
        return {"feature": None, "threshold": None, "left_value": value, "right_value": value}
    _, feature, threshold, left_value, right_value = best
    return {"feature": feature, "threshold": threshold, "left_value": left_value, "right_value": right_value}


def _predict_stump(stump: dict, X: np.ndarray) -> np.ndarray:
    """Predict with a stump returned by `_fit_stump`. Returns shape (n,)."""
    if stump["feature"] is None:
        return np.full(X.shape[0], stump["left_value"])
    goes_left = X[:, stump["feature"]] <= stump["threshold"]
    return np.where(goes_left, stump["left_value"], stump["right_value"])


def in_bag_fraction(idx: np.ndarray, n: int) -> float:
    """Return the fraction of the rows 0, ..., n - 1 that appear in `idx`."""
    return len(np.unique(idx)) / n


# --- Write these in this chapter. ---


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one bootstrap sample of row numbers.

    Draws n numbers from 0, ..., n - 1, each independently and uniformly, with
    replacement. A row can appear more than once, and some rows never appear.

    Args:
        n: Number of rows in the dataset.
        rng: A numpy random Generator to draw from.

    Returns:
        An integer array of shape (n,).
    """
    raise NotImplementedError("bootstrap_indices")


def fit_bagged_stumps(
    X: np.ndarray, y: np.ndarray, n_estimators: int, rng: np.random.Generator
) -> tuple[list[dict], list[np.ndarray]]:
    """Fit one stump on each of n_estimators bootstrap samples.

    For each estimator, draw `idx` with `bootstrap_indices`, then fit
    `_fit_stump(X[idx], y[idx])`. Keep every draw, because the next chapter
    needs to know which rows each stump saw.

    Args:
        X: Features, shape (n, d).
        y: Targets, shape (n,).
        n_estimators: Number of stumps to fit.
        rng: A numpy random Generator, used once per estimator.

    Returns:
        A tuple `(stumps, in_bag_indices)`. `stumps` is a list of
        `n_estimators` dicts from `_fit_stump`. `in_bag_indices[b]` is the
        array `idx` of shape (n,) that stump `b` was fit on.
    """
    raise NotImplementedError("fit_bagged_stumps")


def bagged_predict(stumps: list[dict], X: np.ndarray) -> np.ndarray:
    """Average the predictions of a list of stumps.

    Args:
        stumps: Stumps returned by `fit_bagged_stumps`.
        X: Features, shape (n, d).

    Returns:
        The mean of `_predict_stump(stump, X)` over the stumps, shape (n,).
    """
    raise NotImplementedError("bagged_predict")


def oob_mask(idx: np.ndarray, n: int) -> np.ndarray:
    """Find the rows that one bootstrap draw left out.

    Args:
        idx: The row numbers one stump was fit on, shape (n,), with repeats.
        n: Number of rows in the dataset.

    Returns:
        A boolean array of shape (n,), True for a row that does not appear in
        `idx`.
    """
    raise NotImplementedError("oob_mask")


def oob_predictions(
    X: np.ndarray, stumps: list[dict], in_bag_indices: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every row using only the stumps that did not see it.

    For each row, average `_predict_stump(stump, X)` over the stumps whose draw
    left that row out.

    Args:
        X: Features, shape (n, d).
        stumps: Stumps from `fit_bagged_stumps`.
        in_bag_indices: The matching draws from `fit_bagged_stumps`.

    Returns:
        A tuple `(pred, count)`, both of shape (n,). `count[i]` is the number of
        stumps that missed row i. `pred[i]` is their mean prediction, or `np.nan`
        where `count[i]` is 0.
    """
    raise NotImplementedError("oob_predictions")


def out_of_bag_error(
    X: np.ndarray, y: np.ndarray, stumps: list[dict], in_bag_indices: list[np.ndarray]
) -> float:
    """Compute the out-of-bag mean squared error.

    Average the squared difference between `y` and the out-of-bag prediction,
    over the rows that at least one stump missed.

    Args:
        X: Features, shape (n, d).
        y: Targets, shape (n,).
        stumps: Stumps from `fit_bagged_stumps`.
        in_bag_indices: The matching draws from `fit_bagged_stumps`.

    Returns:
        The out-of-bag mean squared error, as a float.

    Raises:
        ValueError: If no row was out of the bag for any stump.
    """
    raise NotImplementedError("out_of_bag_error")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = bootstrap_indices(10, rng)
    print(f"one draw of 10 rows: {sorted(idx.tolist())}")
    print(f"in-bag fraction of that draw: {in_bag_fraction(idx, 10):.2f}")

    big = np.mean([in_bag_fraction(bootstrap_indices(2000, rng), 2000) for _ in range(20)])
    print(f"in-bag fraction at n = 2000, averaged over 20 draws: {big:.3f} (1 - 1/e is 0.632)")

    X = np.sort(rng.uniform(0, 1, size=(40, 1)), axis=0)
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.normal(size=40)
    stumps, in_bag = fit_bagged_stumps(X, y, n_estimators=15, rng=rng)
    pred = bagged_predict(stumps, X)
    print(f"fitted {len(stumps)} stumps, each on {len(in_bag[0])} drawn rows")
    print(f"single-stump training MSE: {np.mean((_predict_stump(stumps[0], X) - y) ** 2):.3f}")
    print(f"bagged training MSE: {np.mean((pred - y) ** 2):.3f}")

    n = 500
    X = rng.normal(size=(n, 3))
    true_f = lambda X: np.maximum(X[:, 0], 0.0) - 0.5 * X[:, 1]
    y = true_f(X) + 0.3 * rng.normal(size=n)
    X_test = rng.normal(size=(n, 3))
    y_test = true_f(X_test) + 0.3 * rng.normal(size=n)
    stumps, in_bag = fit_bagged_stumps(X, y, n_estimators=80, rng=rng)
    pred, count = oob_predictions(X, stumps, in_bag)
    print(f"each row was out of the bag for {count.mean():.1f} of 80 stumps on average")
    print(f"training error (all stumps, every row): {np.mean((bagged_predict(stumps, X) - y) ** 2):.4f}")
    print(f"out-of-bag error: {out_of_bag_error(X, y, stumps, in_bag):.4f}")
    print(f"test error on {n} fresh rows: {np.mean((bagged_predict(stumps, X_test) - y_test) ** 2):.4f}")
