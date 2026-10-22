"""Reference solution for the bootstrap and bagging exercise."""

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
