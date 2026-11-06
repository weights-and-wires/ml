"""Build a learning curve and read it.

You write three functions, in this order:

1. `validation_split`: one fixed shuffle that sets aside a validation set.
2. `learning_curve`: training and validation error at growing training sizes.
3. `diagnose_curve`: turn the two curves into 'underfit', 'overfit', or 'good'.

Wherever `fit` and `predict` appear as arguments, they follow the pattern
`model = fit(X_train, y_train)` and `y_pred = predict(model, X_test)`, so the
same functions work for any model.
Shapes: X is (n, d), one row per house; y is (n,).
"""

import numpy as np


def validation_split(n: int, val_fraction: float, seed: int):
    """Shuffle the row numbers once and set aside a validation set.

    Args:
        n: Number of rows.
        val_fraction: Fraction of the rows to hold out for validation.
        seed: Seed for `np.random.default_rng`, so the split repeats exactly.

    Returns:
        A tuple `(val_idx, pool_idx)` of integer arrays. `val_idx` holds
        `int(round(val_fraction * n))` rows, `pool_idx` holds the rest in
        shuffled order, and together they cover `range(n)` once.
    """
    order = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_fraction * n))
    return order[:n_val], order[n_val:]


def learning_curve(
    fit,
    predict,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes,
    val_fraction: float = 0.25,
    seed: int = 0,
):
    """Compute training and validation error at growing training sizes.

    Call `validation_split` once. For each size, fit on the first `size`
    rows of the pool, so each training set contains the previous one, and
    record the mean squared error on those training rows and on the fixed
    validation rows.

    Args:
        fit: Callable `fit(X_train, y_train) -> model`.
        predict: Callable `predict(model, X_test) -> y_pred`.
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        train_sizes: Sequence of training-set sizes, each at most the pool size.
        val_fraction: Fraction of the rows held out for validation.
        seed: Seed for the split.

    Returns:
        A tuple `(train_errors, val_errors)`, each of shape (len(train_sizes),).
    """
    val_idx, pool_idx = validation_split(X.shape[0], val_fraction, seed)
    train_errors = np.zeros(len(train_sizes))
    val_errors = np.zeros(len(train_sizes))
    for j, size in enumerate(train_sizes):
        rows = pool_idx[:size]
        model = fit(X[rows], y[rows])
        train_errors[j] = np.mean((predict(model, X[rows]) - y[rows]) ** 2)
        val_errors[j] = np.mean((predict(model, X[val_idx]) - y[val_idx]) ** 2)
    return train_errors, val_errors


def diagnose_curve(train_errors, val_errors, high_threshold: float, gap_threshold: float) -> str:
    """Read a learning curve at its largest training size.

    The gap is `val_errors[-1] - train_errors[-1]`.

    Args:
        train_errors: Training error at each size, shape (m,).
        val_errors: Validation error at each size, shape (m,).
        high_threshold: Validation error above this counts as too high.
        gap_threshold: A gap above this counts as wide.

    Returns:
        'overfit' if the gap is above `gap_threshold`; otherwise 'underfit'
        if the last validation error is above `high_threshold`; otherwise 'good'.
    """
    gap = val_errors[-1] - train_errors[-1]
    if gap > gap_threshold:
        return "overfit"
    if val_errors[-1] > high_threshold:
        return "underfit"
    return "good"


if __name__ == "__main__":
    # Sixty houses, one measurement each (floor area), and a price that curves.
    rng = np.random.default_rng(3)
    n = 60
    area = rng.uniform(-1, 1, n)
    price = np.sin(3 * area) + 0.5 * rng.normal(size=n)

    def make_fit(degree):
        def fit(x, y):
            F = np.vander(x[:, 0], degree + 1, increasing=True)
            return np.linalg.solve(F.T @ F + 1e-3 * np.eye(degree + 1), F.T @ y)

        def predict(w, x):
            return np.vander(x[:, 0], degree + 1, increasing=True) @ w

        return fit, predict

    sizes = [10, 14, 18, 24, 30, 38, 45]
    for name, degree in (("straight line", 1), ("degree 8 curve", 8)):
        fit, predict = make_fit(degree)
        # One shuffle gives a noisy curve, so take the median over 60 shuffles.
        runs = [learning_curve(fit, predict, area[:, None], price, sizes, 0.25, seed=s) for s in range(60)]
        train = np.median([r[0] for r in runs], axis=0)
        val = np.median([r[1] for r in runs], axis=0)
        verdict = diagnose_curve(train, val, high_threshold=0.4, gap_threshold=0.08)
        print(f"{name}: train {np.round(train, 2)}")
        print(f"{' ' * len(name)}  validation {np.round(val, 2)} -> {verdict}")
