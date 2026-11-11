"""Feature pipelines and data leakage.

You write these pieces, in this order:

1. `Standardiser`: `fit` stores each column's mean and standard deviation from the
   training rows, and `transform` uses only those stored numbers.
2. `Imputer`: `fit` stores a per-column mean or median of the observed entries, and
   `transform` fills every NaN with it.
3. `one_hot`: one column per fitted category, plus a last column for unseen values.
4. `TargetEncoder`: replaces a category by a smoothed mean of the target.
5. `target_encode_oof`: encodes every row with a table learned from the other folds.
6. `select_top_k`: keeps the k columns most correlated with the target.
7. `cv_score`: refits a whole pipeline inside every fold and scores the held-out fold.
8. `leaky_cv_accuracy` and `safe_cv_accuracy`: the same selection done before and
   inside the folds. On pure noise only one of them tells the truth.
9. `count_cross_fold_duplicates`: how many held-out rows have an exact twin in training.

Two helpers are provided: `k_fold_indices` from the cross-validation chapter, and
`fit_centroid`, a nearest-centroid classifier for labels 0 and 1.
Shapes: X is (n, d), one row per example; y is (n,).
"""

from __future__ import annotations

import numpy as np


# Provided, not for you to write.
def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """k sorted, disjoint folds of the row numbers 0..n-1, as in the previous chapter."""
    return [np.sort(part) for part in np.array_split(rng.permutation(n), k)]


# Provided, not for you to write.
def fit_centroid(X: np.ndarray, y: np.ndarray):
    """Fit a nearest-centroid classifier for labels 0 and 1.

    Returns a function `predict(X_new)` that gives an int array of 0s and 1s.
    """
    c0, c1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)

    def predict(X_new: np.ndarray) -> np.ndarray:
        closer_to_1 = ((X_new - c1) ** 2).sum(axis=1) < ((X_new - c0) ** 2).sum(axis=1)
        return closer_to_1.astype(int)

    return predict


class Standardiser:
    """Centre and scale each column with statistics stored by `fit`."""

    def fit(self, X: np.ndarray) -> "Standardiser":
        """Store `self.mean_` and `self.std_`, each of shape (d,), and return self.

        `std_` is the population standard deviation (divide by n). Where a column
        is constant, store 1.0 instead of 0.0 so the division is safe.
        """
        raise NotImplementedError("Standardiser.fit")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return (X - mean_) / std_, shape (m, d), using only the stored arrays."""
        raise NotImplementedError("Standardiser.transform")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then transform X."""
        return self.fit(X).transform(X)


class Imputer:
    """Fill missing entries (NaN) with a per-column statistic stored by `fit`."""

    def __init__(self, strategy: str = "mean"):
        if strategy not in ("mean", "median"):
            raise ValueError(f"strategy must be 'mean' or 'median', got {strategy!r}")
        self.strategy = strategy

    def fit(self, X: np.ndarray) -> "Imputer":
        """Store `self.fill_`, shape (d,): the mean or median of each column's
        observed (non-NaN) entries. A column with no observed entry gets 0.0.
        Return self.
        """
        raise NotImplementedError("Imputer.fit")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of X, shape (m, d), with each NaN replaced by that column's
        stored fill. Do not change the input array.
        """
        raise NotImplementedError("Imputer.transform")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then transform X."""
        return self.fit(X).transform(X)


def one_hot(fit_categories, values) -> np.ndarray:
    """One-hot encode `values` against the categories seen at fit time.

    Args:
        fit_categories: List of m distinct categories. Column j stands for the j-th.
        values: List of n category values to encode. Some may not be in
            `fit_categories`.

    Returns:
        A float array of shape (n, m + 1). Column m is the unknown bucket: it holds
        a 1 for every value that is not in `fit_categories`. Each row sums to 1.
    """
    raise NotImplementedError("one_hot")


class TargetEncoder:
    """Replace a category by a smoothed mean of the target, learned in `fit`."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def fit(self, categories, y: np.ndarray) -> "TargetEncoder":
        """Learn `self.prior_` (the mean of y) and `self.table_`, a dict from each
        category to (sum of its y + smoothing * prior) / (its count + smoothing).
        Return self.
        """
        raise NotImplementedError("TargetEncoder.fit")

    def transform(self, categories) -> np.ndarray:
        """Return the encoding of each category, shape (n,). A category that was not
        seen in `fit` gets `prior_`.
        """
        raise NotImplementedError("TargetEncoder.transform")


def target_encode_oof(categories, y: np.ndarray, folds, smoothing: float = 1.0) -> np.ndarray:
    """Encode every row with a table learned from the other folds only.

    For each test fold, fit a `TargetEncoder` on the rows outside that fold and
    encode the fold's rows with it. A row's own target never reaches its encoding.

    Args:
        categories: Sequence of n category values.
        y: Targets, shape (n,).
        folds: List of index arrays that partition range(n).
        smoothing: Pseudo-count that pulls each category toward the prior.

    Returns:
        The out-of-fold encoding, shape (n,).
    """
    raise NotImplementedError("target_encode_oof")


def select_top_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Pick the k columns whose absolute correlation with y is largest.

    Args:
        X: Features, shape (n, d).
        y: Targets, shape (n,).
        k: How many columns to keep.

    Returns:
        The k chosen column numbers, sorted ascending, shape (k,).
    """
    raise NotImplementedError("select_top_k")


def cv_score(pipeline_fit, X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Score a pipeline on every held-out fold, refitting it from scratch each time.

    Args:
        pipeline_fit: Callable `pipeline_fit(X_train, y_train) -> predict`, where
            `predict(X_test)` returns predicted labels. Everything the pipeline
            learns, including preprocessing, happens inside this call.
        X: Features, shape (n, d).
        y: Labels, shape (n,).
        folds: List of k index arrays that partition range(n).

    Returns:
        The accuracy of each fold, shape (k,).
    """
    raise NotImplementedError("cv_score")


def leaky_cv_accuracy(X: np.ndarray, y: np.ndarray, n_keep: int, folds) -> float:
    """Select `n_keep` columns with `select_top_k` on ALL rows, then cross-validate.

    Classify with `fit_centroid` on the kept columns. This is the wrong order, and
    you write it to see what it does. Return the mean fold accuracy.
    """
    raise NotImplementedError("leaky_cv_accuracy")


def safe_cv_accuracy(X: np.ndarray, y: np.ndarray, n_keep: int, folds) -> float:
    """Select `n_keep` columns inside each fold, using that fold's training rows only.

    Classify with `fit_centroid` on the kept columns. Return the mean fold accuracy.
    """
    raise NotImplementedError("safe_cv_accuracy")


def count_cross_fold_duplicates(X: np.ndarray, folds) -> int:
    """Count held-out rows that have an identical row among the training rows.

    For each fold, the training rows are every row outside it. Sum the count over
    folds. A row that appears twice in the data, once in each side of a split,
    counts once per round in which it is held out opposite its twin.
    """
    raise NotImplementedError("count_cross_fold_duplicates")


def demo_pipeline() -> None:
    train = np.array([[2.0, 1.0], [4.0, np.nan], [6.0, 9.0], [8.0, 2.0]])
    test = np.array([[10.0, np.nan]])
    imp = Imputer("median").fit(train)
    std = Standardiser().fit(imp.transform(train))
    print("stored mean:", std.mean_, "stored std:", np.round(std.std_, 4))
    print("test row after the pipeline:", np.round(std.transform(imp.transform(test)), 4))
    print("one_hot:", one_hot(["red", "green", "blue"], ["green", "purple"]).tolist())


def demo_leakage() -> None:
    n, d = 50, 500
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, size=n)  # pure noise: no column knows anything about y
    folds = k_fold_indices(n, 5, np.random.default_rng(1))
    print(f"leaky accuracy on noise: {leaky_cv_accuracy(X, y, 10, folds):.3f}")
    print(f"safe accuracy on noise:  {safe_cv_accuracy(X, y, 10, folds):.3f}  (chance is 0.5)")


if __name__ == "__main__":
    demo_pipeline()
    print()
    demo_leakage()
