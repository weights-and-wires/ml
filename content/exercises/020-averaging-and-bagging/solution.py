"""Averaging models, and bagging them: why an average cuts noise, and how to score it for free.

Each model's prediction at one spot is a shared part plus a private part.
Averaging B models cancels the private parts and leaves the shared part, so the
variance of the average has a floor. The bootstrap then supplies the different
training sets, and the rows each draw misses score the ensemble without holding
any data back. You write nine functions, in this order:

1. `bagging_variance`: the variance of an average of B models.
2. `correlated_estimators`: simulate models with a chosen variance and correlation.
3. `trees_needed`: how many models bring the variance close to the floor.
4. `bootstrap_indices`: one draw of n row numbers, with replacement.
5. `fit_bagged_stumps`: one stump per draw, keeping the draws.
6. `bagged_predict`: the mean of the stumps' predictions.
7. `oob_mask`: which rows one draw left out.
8. `oob_predictions`: for each row, the mean over the stumps that missed it.
9. `out_of_bag_error`: the mean squared error on those held-out predictions.

`_fit_stump`, `_predict_stump`, and `in_bag_fraction` are written for you.
"""

import math

import numpy as np


def bagging_variance(rho: float, sigma2: float, B: int) -> float:
    """Compute the variance of the average of B models.

    Every model has variance `sigma2`, and every pair of models has
    correlation `rho`. The shared part, `rho * sigma2`, is not divided by `B`.
    The private part, `(1 - rho) * sigma2`, is divided by `B`.

    Args:
        rho: Correlation between any two models, from 0 to 1.
        sigma2: Variance of a single model.
        B: Number of models averaged.

    Returns:
        The variance of the average, as a float.
    """
    return float(rho * sigma2 + (1.0 - rho) * sigma2 / B)


def correlated_estimators(
    rho: float, sigma2: float, B: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate B models whose predictions share a part and keep a private part.

    Entry `[i, b]` is `sqrt(rho * sigma2) * shared[i] + sqrt((1 - rho) * sigma2) * private[i, b]`,
    where `shared` is one standard normal draw per row, `private` is an
    independent standard normal draw per entry, and a row is one redraw of the
    training set. Then every column has variance `sigma2` and every pair of
    columns has correlation `rho`.

    Args:
        rho: Correlation between any two columns, from 0 to 1.
        sigma2: Variance of each column.
        B: Number of models, the number of columns.
        n_draws: Number of redraws, the number of rows.
        rng: A numpy random Generator to draw from.

    Returns:
        An array of shape (n_draws, B).
    """
    shared = rng.normal(size=(n_draws, 1))
    private = rng.normal(size=(n_draws, B))
    return np.sqrt(rho * sigma2) * shared + np.sqrt((1.0 - rho) * sigma2) * private


def trees_needed(rho: float, sigma2: float, tol: float) -> int:
    """Find the smallest B whose variance is within a fraction `tol` of the floor.

    The floor is `rho * sigma2`. The variance with `B` models is within a
    fraction `tol` of it when `(1 - rho) * sigma2 / B <= tol * rho * sigma2`.

    Args:
        rho: Correlation between any two models, from 0 to 1.
        sigma2: Variance of a single model.
        tol: Allowed gap above the floor, as a fraction of the floor.

    Returns:
        The smallest integer B, and at least 1.

    Raises:
        ValueError: If `rho` is 0, because then the floor is 0 and no finite B
            gets within a fraction of it.
    """
    if rho == 0:
        raise ValueError("rho is 0, so the floor is 0 and no finite B gets within a fraction of it")
    return max(1, math.ceil((1.0 - rho) / (tol * rho)))


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
    return rng.integers(0, n, size=n)


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
    n = X.shape[0]
    stumps = []
    in_bag_indices = []
    for _ in range(n_estimators):
        idx = bootstrap_indices(n, rng)
        stumps.append(_fit_stump(X[idx], y[idx]))
        in_bag_indices.append(idx)
    return stumps, in_bag_indices


def bagged_predict(stumps: list[dict], X: np.ndarray) -> np.ndarray:
    """Average the predictions of a list of stumps.

    Args:
        stumps: Stumps returned by `fit_bagged_stumps`.
        X: Features, shape (n, d).

    Returns:
        The mean of `_predict_stump(stump, X)` over the stumps, shape (n,).
    """
    return np.mean([_predict_stump(stump, X) for stump in stumps], axis=0)


def oob_mask(idx: np.ndarray, n: int) -> np.ndarray:
    """Find the rows that one bootstrap draw left out.

    Args:
        idx: The row numbers one stump was fit on, shape (n,), with repeats.
        n: Number of rows in the dataset.

    Returns:
        A boolean array of shape (n,), True for a row that does not appear in
        `idx`.
    """
    in_bag = np.zeros(n, dtype=bool)
    in_bag[idx] = True
    return ~in_bag


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
    n = X.shape[0]
    total = np.zeros(n)
    count = np.zeros(n)
    for stump, idx in zip(stumps, in_bag_indices):
        missed = oob_mask(idx, n)
        total[missed] += _predict_stump(stump, X[missed])
        count[missed] += 1
    pred = np.full(n, np.nan)
    seen = count > 0
    pred[seen] = total[seen] / count[seen]
    return pred, count


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
    pred, count = oob_predictions(X, stumps, in_bag_indices)
    covered = count > 0
    if not np.any(covered):
        raise ValueError("every row is in every bag, so there is nothing to score; use more stumps")
    return float(np.mean((pred[covered] - y[covered]) ** 2))


if __name__ == "__main__":
    print(f"sigma2=4, rho=0.5: B=1 gives {bagging_variance(0.5, 4.0, 1):.2f}, B=2 gives {bagging_variance(0.5, 4.0, 2):.2f}")
    print(f"rho=0.3, sigma2=1: B=50 gives {bagging_variance(0.3, 1.0, 50):.4f}, B=5000 gives {bagging_variance(0.3, 1.0, 5000):.5f}")

    rng = np.random.default_rng(0)
    draws = correlated_estimators(0.3, 1.0, 50, 20000, rng)
    print(f"simulated variance of the average of 50 models: {draws.mean(axis=1).var():.4f}")

    print(f"models needed to get within 5% of the floor at rho=0.3: {trees_needed(0.3, 1.0, 0.05)}")

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
