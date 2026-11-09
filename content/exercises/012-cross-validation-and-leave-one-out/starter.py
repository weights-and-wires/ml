"""Cross-validation and the leave-one-out shortcut.

You write six functions, in this order:

1. `k_fold_indices`: shuffle the row numbers and cut them into k folds.
2. `cross_val_score`: fit on k - 1 folds, score the held-out fold, k times.
3. `best_lambda_by_cv`: score every ridge penalty on the same folds and pick the best.
4. `hat_matrix`: the n-by-n matrix H that turns the targets into ridge fitted values.
5. `leverages`: how much each point controls its own fitted value, the diagonal of H.
6. `loo_linear_smoother`: the leave-one-out mean squared error from one fit.

Wherever `fit` and `predict` appear as arguments, they follow the pattern
`model = fit(X_train, y_train)` and `y_pred = predict(model, X_test)`, so the
same functions work for any model, not only ridge regression.
A seventh function, `loo_by_refitting`, is provided. It does the same job as
`loo_linear_smoother` the slow way, with n refits, so you can compare them.
Shapes: X is (n, d), one row per point; y is (n,).
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
    raise NotImplementedError("k_fold_indices")


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
    raise NotImplementedError("cross_val_score")


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
    raise NotImplementedError("best_lambda_by_cv")


def hat_matrix(X: np.ndarray, lam: float) -> np.ndarray:
    """Build H = X (X^T X + lam I)^{-1} X^T for ridge regression.

    H "puts a hat on y": `H @ y` is the vector of fitted values of the ridge
    fit on all n points.

    Args:
        X: Design matrix, shape (n, d).
        lam: Nonnegative ridge penalty.

    Returns:
        The hat matrix, shape (n, n).
    """
    raise NotImplementedError("hat_matrix")


def leverages(X: np.ndarray, lam: float) -> np.ndarray:
    """Return how much each point controls its own fitted value.

    Args:
        X: Design matrix, shape (n, d).
        lam: Nonnegative ridge penalty.

    Returns:
        The diagonal of the hat matrix, shape (n,).
    """
    raise NotImplementedError("leverages")


def loo_linear_smoother(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Compute the leave-one-out mean squared error of ridge from one fit.

    With r = y - H y the in-sample residuals and h the leverages, the
    leave-one-out residual of point i is r_i / (1 - h_i).

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Nonnegative ridge penalty.

    Returns:
        The mean of the squared leave-one-out residuals, one float.
    """
    raise NotImplementedError("loo_linear_smoother")


# Provided, not for you to write: the slow way, with n separate fits.
def loo_by_refitting(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Leave-one-out mean squared error by fitting ridge n times."""
    n, d = X.shape
    errors = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        w = np.linalg.solve(X[keep].T @ X[keep] + lam * np.eye(d), X[keep].T @ y[keep])
        errors[i] = (y[i] - X[i] @ w) ** 2
    return float(np.mean(errors))


def demo_cross_validation() -> None:
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


def demo_leave_one_out() -> None:
    # The chapter's hand example: inputs 1, 2, 3, targets 1, 2, 2, penalty 1.
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 2.0])
    print("leverages:", np.round(leverages(X, 1.0), 4), "(want 1/15, 4/15, 9/15)")
    print("in-sample residuals:", np.round(y - hat_matrix(X, 1.0) @ y, 4))
    print(f"leave-one-out MSE, shortcut:  {loo_linear_smoother(X, y, 1.0):.4f}")
    print(f"leave-one-out MSE, refitting: {loo_by_refitting(X, y, 1.0):.4f}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = X @ rng.normal(size=6) + rng.normal(size=200)
    for lam in (0.1, 1.0, 10.0):
        print(f"lam={lam:<5} shortcut {loo_linear_smoother(X, y, lam):.6f}"
              f"  refitting {loo_by_refitting(X, y, lam):.6f}")


if __name__ == "__main__":
    demo_cross_validation()
    print()
    demo_leave_one_out()
