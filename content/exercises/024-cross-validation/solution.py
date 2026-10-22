"""Pick a ridge penalty with k-fold cross-validation.

You write three functions, in this order:

1. `k_fold_indices`: shuffle the row numbers and cut them into k folds.
2. `cross_val_score`: fit on k - 1 folds, score the held-out fold, k times.
3. `best_lambda_by_cv`: score every ridge penalty on the same folds and pick the best.

Wherever `fit` and `predict` appear as arguments, they follow the pattern
`model = fit(X_train, y_train)` and `y_pred = predict(model, X_test)`, so the
same functions work for any model, not only ridge regression.
Shapes: X is (n, d), one row per house; y is (n,).
"""

import numpy as np


def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Partition {0, ..., n - 1} into k folds of nearly equal size.

    Args:
        n: Number of data points.
        k: Number of folds, 2 <= k <= n.
        rng: Source of randomness for the shuffle before splitting.

    Returns:
        A list of k sorted, disjoint integer arrays whose union is
        `range(n)`. Sizes differ by at most one across folds.
    """
    order = rng.permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


def cross_val_score(fit, predict, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Compute the held-out mean squared error of each of k folds.

    Build the folds with `k_fold_indices` and a generator created inside this
    function with `np.random.default_rng(0)`, so every call on the same n
    and k returns the same folds. For each fold, fit on the other k - 1 folds
    and measure the mean squared error on the held-out fold.

    Args:
        fit: Callable `fit(X_train, y_train) -> model`.
        predict: Callable `predict(model, X_test) -> y_pred`.
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        k: Number of folds.

    Returns:
        An array of shape (k,), the held-out mean squared error of each fold.
    """
    n = X.shape[0]
    folds = k_fold_indices(n, k, np.random.default_rng(0))
    scores = np.zeros(k)
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        model = fit(X[train], y[train])
        scores[f] = np.mean((predict(model, X[test]) - y[test]) ** 2)
    return scores


def best_lambda_by_cv(X: np.ndarray, y: np.ndarray, lambdas, k: int = 5):
    """Choose the ridge penalty with the lowest cross-validated error.

    For each penalty, fit ridge regression with
    w = (X^T X + lam I)^{-1} X^T y on the training folds, score it with
    `cross_val_score`, and average the k fold scores.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lambdas: Sequence of m nonnegative penalties to compare.
        k: Number of folds.

    Returns:
        A tuple `(best_lambda, cv_means)`: the penalty with the smallest mean
        fold error, and an array of shape (m,) with every penalty's mean.
    """
    d = X.shape[1]
    means = np.zeros(len(lambdas))
    for j, lam in enumerate(lambdas):

        def fit(Xt, yt, lam=lam):
            return np.linalg.solve(Xt.T @ Xt + lam * np.eye(d), Xt.T @ yt)

        means[j] = cross_val_score(fit, lambda w, Xs: Xs @ w, X, y, k).mean()
    return lambdas[int(np.argmin(means))], means


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n, d = 60, 30
    Z = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = Z @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = 0.3 * rng.normal(size=d)
    y = X @ w_true + rng.normal(size=n)

    lambdas = np.logspace(-2, 3, 20)
    best, cv_means = best_lambda_by_cv(X, y, lambdas, k=5)

    def ridge(Xt, yt, lam):
        return np.linalg.solve(Xt.T @ Xt + lam * np.eye(d), Xt.T @ yt)

    train_errors = [np.mean((X @ ridge(X, y, l) - y) ** 2) for l in lambdas]
    print(f"penalty with the lowest training error: {lambdas[int(np.argmin(train_errors))]:.3g}")
    print(f"penalty with the lowest 5-fold error:   {best:.3g}")
    print(f"5-fold error at that penalty:           {cv_means.min():.3f}")
