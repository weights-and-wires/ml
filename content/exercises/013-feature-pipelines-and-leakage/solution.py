"""Reference solution for the feature pipelines and data leakage exercise."""

from __future__ import annotations

import numpy as np


def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Provided from the cross-validation chapter: k sorted, disjoint folds."""
    return [np.sort(part) for part in np.array_split(rng.permutation(n), k)]


def fit_centroid(X: np.ndarray, y: np.ndarray):
    """Provided: a nearest-centroid classifier for labels 0 and 1."""
    c0, c1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)

    def predict(X_new: np.ndarray) -> np.ndarray:
        closer_to_1 = ((X_new - c1) ** 2).sum(axis=1) < ((X_new - c0) ** 2).sum(axis=1)
        return closer_to_1.astype(int)

    return predict


class Standardiser:
    """Centre and scale each column with statistics stored by `fit`."""

    def fit(self, X: np.ndarray) -> "Standardiser":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class Imputer:
    """Fill missing entries (NaN) with a per-column statistic stored by `fit`."""

    def __init__(self, strategy: str = "mean"):
        if strategy not in ("mean", "median"):
            raise ValueError(f"strategy must be 'mean' or 'median', got {strategy!r}")
        self.strategy = strategy

    def fit(self, X: np.ndarray) -> "Imputer":
        X = np.asarray(X, dtype=float)
        stat = np.nanmean if self.strategy == "mean" else np.nanmedian
        fill = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            seen = X[~np.isnan(X[:, j]), j]
            fill[j] = stat(seen) if seen.size else 0.0
        self.fill_ = fill
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        rows, cols = np.nonzero(np.isnan(X))
        X[rows, cols] = self.fill_[cols]
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def one_hot(fit_categories, values) -> np.ndarray:
    """One column per fitted category, plus a last column for unseen values."""
    index = {c: j for j, c in enumerate(fit_categories)}
    out = np.zeros((len(values), len(index) + 1))
    for i, v in enumerate(values):
        out[i, index.get(v, len(index))] = 1.0
    return out


class TargetEncoder:
    """Replace a category by a smoothed mean of the target, learned in `fit`."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def fit(self, categories, y: np.ndarray) -> "TargetEncoder":
        y = np.asarray(y, dtype=float)
        categories = np.asarray(categories)
        self.prior_ = float(y.mean())
        self.table_ = {}
        for c in np.unique(categories):
            rows = categories == c
            total, count = y[rows].sum(), rows.sum()
            self.table_[c.item()] = float(
                (total + self.smoothing * self.prior_) / (count + self.smoothing)
            )
        return self

    def transform(self, categories) -> np.ndarray:
        return np.array([self.table_.get(c, self.prior_) for c in np.asarray(categories).tolist()])


def target_encode_oof(categories, y: np.ndarray, folds, smoothing: float = 1.0) -> np.ndarray:
    """Encode every row with a table learned from the other folds only."""
    categories = np.asarray(categories)
    y = np.asarray(y, dtype=float)
    out = np.empty(len(y))
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        encoder = TargetEncoder(smoothing).fit(categories[train], y[train])
        out[test] = encoder.transform(categories[test])
    return out


def select_top_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k columns whose absolute correlation with y is largest."""
    Xc = X - X.mean(axis=0)
    yc = np.asarray(y, dtype=float) - np.mean(y)
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.abs(Xc.T @ yc) / np.where(denom == 0.0, 1.0, denom)
    return np.sort(np.argsort(-corr, kind="stable")[:k])


def cv_score(pipeline_fit, X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Accuracy on each held-out fold, refitting the whole pipeline every time."""
    scores = np.empty(len(folds))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        predict = pipeline_fit(X[train], y[train])
        scores[f] = np.mean(predict(X[test]) == y[test])
    return scores


def leaky_cv_accuracy(X: np.ndarray, y: np.ndarray, n_keep: int, folds) -> float:
    """Select columns on all rows, then cross-validate. Leaks the held-out labels."""
    keep = select_top_k(X, y, n_keep)

    def pipeline_fit(X_train, y_train):
        predict = fit_centroid(X_train[:, keep], y_train)
        return lambda X_new: predict(X_new[:, keep])

    return float(cv_score(pipeline_fit, X, y, folds).mean())


def safe_cv_accuracy(X: np.ndarray, y: np.ndarray, n_keep: int, folds) -> float:
    """Select columns inside each fold, on the training rows only."""

    def pipeline_fit(X_train, y_train):
        keep = select_top_k(X_train, y_train, n_keep)
        predict = fit_centroid(X_train[:, keep], y_train)
        return lambda X_new: predict(X_new[:, keep])

    return float(cv_score(pipeline_fit, X, y, folds).mean())


def count_cross_fold_duplicates(X: np.ndarray, folds) -> int:
    """Held-out rows that also appear, exactly, in the training rows of their round."""
    X = np.asarray(X)
    total = 0
    for test in folds:
        train = np.setdiff1d(np.arange(len(X)), test)
        seen = {row.tobytes() for row in np.ascontiguousarray(X[train])}
        total += sum(row.tobytes() in seen for row in np.ascontiguousarray(X[test]))
    return int(total)


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
