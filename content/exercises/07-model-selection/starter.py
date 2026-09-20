"""Cross-validation, the leave-one-out shortcut, learning curves, and Rademacher
complexity by sampling.

Implement five functions. `fit` and `predict`, wherever they appear as
arguments, follow the convention `model = fit(X_train, y_train)` and
`y_pred = predict(model, X_test)`, so the same `cross_val_score` and
`learning_curve` work for any estimator, not only ridge regression.
"""

import numpy as np


def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Partition {0, ..., n - 1} into k folds of nearly equal size.

    Args:
        n: Number of data points.
        k: Number of folds, 2 <= k <= n.
        rng: Source of randomness for the shuffle before splitting.

    Returns:
        A list of k sorted, disjoint arrays of indices whose union is
        `range(n)`. Sizes differ by at most one across folds.
    """
    raise NotImplementedError("k_fold_indices")


def cross_val_score(fit, predict, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Compute the k-fold cross-validated mean squared error.

    Build folds with `k_fold_indices` using a fixed, internal random seed, so
    the same call always returns the same folds. For each fold, fit on the
    other k - 1 folds and evaluate the mean squared error on the held-out
    fold.

    Args:
        fit: Callable `fit(X_train, y_train) -> model`.
        predict: Callable `predict(model, X_test) -> y_pred`.
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        k: Number of folds.

    Returns:
        An array of shape (k,) holding the held-out mean squared error of
        each fold.
    """
    raise NotImplementedError("cross_val_score")


def loo_linear_smoother(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Compute the leave-one-out mean squared error of ridge regression.

    Use the hat-matrix shortcut rather than refitting n times: fit once on
    the full data to get the hat matrix H = X (X^T X + lam I)^{-1} X^T and
    the in-sample residual r = y - H y, then each leave-one-out residual is
    r_i / (1 - H_ii).

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Nonnegative ridge penalty.

    Returns:
        The mean of the squared leave-one-out residuals.
    """
    raise NotImplementedError("loo_linear_smoother")


def learning_curve(
    fit,
    predict,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute training and validation error at a sequence of training sizes.

    Set aside a validation set of `round(val_fraction * n)` points using
    `seed`, drawn once and reused for every entry in `train_sizes`. For each
    size, fit on that many points from the remaining pool (the first `size`
    of a single fixed shuffle of the pool, so training sets nest) and record
    the mean squared error on that training subset and on the validation set.

    Args:
        fit: Callable `fit(X_train, y_train) -> model`.
        predict: Callable `predict(model, X_test) -> y_pred`.
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        train_sizes: A sequence of training-set sizes to evaluate.
        val_fraction: Fraction of the data held out for validation.
        seed: Random seed for the validation split and the pool order.

    Returns:
        A tuple `(train_errors, val_errors)`, each an array with one entry
        per element of `train_sizes`.
    """
    raise NotImplementedError("learning_curve")


def empirical_rademacher(
    losses_matrix: np.ndarray, rng: np.random.Generator, num_draws: int = 2000
) -> float:
    """Estimate the empirical Rademacher complexity of a finite function class.

    `losses_matrix[j, i]` is f_j(z_i) for hypothesis j and sample point i.
    Draw `num_draws` independent Rademacher vectors sigma in {-1, 1}^n, and
    for each one compute the supremum over hypotheses of
    (1 / n) * sigma . f_j, then average over draws.

    Args:
        losses_matrix: Array of shape (num_hypotheses, n).
        rng: Source of randomness for the Rademacher draws.
        num_draws: Number of Monte Carlo draws.

    Returns:
        The Monte Carlo estimate of the empirical Rademacher complexity.
    """
    raise NotImplementedError("empirical_rademacher")


if __name__ == "__main__":
    def ridge_fit(X, y, lam):
        d = X.shape[1]
        return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)

    def ridge_predict(w, X):
        return X @ w

    rng = np.random.default_rng(0)
    n, d = 60, 30
    Z = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = Z @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = 0.3 * rng.normal(size=d)
    y = X @ w_true + rng.normal(size=n)

    lambdas = np.logspace(-2, 3, 15)
    cv_curve = [
        cross_val_score(lambda Xt, yt, l=l: ridge_fit(Xt, yt, l), ridge_predict, X, y, k=5).mean()
        for l in lambdas
    ]
    best = lambdas[int(np.argmin(cv_curve))]
    print(f"cross-validated ridge penalty: {best:.3g}")

    loo_mse = [loo_linear_smoother(X, y, l) for l in lambdas]
    best_loo = lambdas[int(np.argmin(loo_mse))]
    print(f"leave-one-out ridge penalty:   {best_loo:.3g}")
