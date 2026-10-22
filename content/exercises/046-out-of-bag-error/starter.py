"""Out-of-bag error: a free test set.

Each bootstrap sample misses about a third of the rows. Scoring each row with
only the trees that missed it gives an honest held-out error without setting
any data aside. You write three functions, in this order:

1. `oob_mask`: which rows one draw left out.
2. `oob_predictions`: for each row, the mean over the trees that missed it.
3. `out_of_bag_error`: the mean squared error of those predictions.

The functions from the previous chapter are already here, solved.
"""

import numpy as np


# --- Provided helpers: a depth-1 regression tree. ---


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


# --- Written in the chapter "Bootstrap samples and bagging". Provided, already solved. ---


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def fit_bagged_stumps(
    X: np.ndarray, y: np.ndarray, n_estimators: int, rng: np.random.Generator
) -> tuple[list[dict], list[np.ndarray]]:
    n = X.shape[0]
    stumps = []
    in_bag_indices = []
    for _ in range(n_estimators):
        idx = bootstrap_indices(n, rng)
        stumps.append(_fit_stump(X[idx], y[idx]))
        in_bag_indices.append(idx)
    return stumps, in_bag_indices


def bagged_predict(stumps: list[dict], X: np.ndarray) -> np.ndarray:
    return np.mean([_predict_stump(stump, X) for stump in stumps], axis=0)


# --- Write these in this chapter. ---


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
