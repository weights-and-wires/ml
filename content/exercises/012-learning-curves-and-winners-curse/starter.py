"""Learning curves and the winner's curse.

You write six functions, in this order:

1. `validation_split`: one fixed shuffle that sets aside a validation set.
2. `learning_curve`: training and validation error at growing training sizes.
3. `diagnose_curve`: turn the two curves into 'underfit', 'overfit', or 'good'.
4. `best_of_m_gap`: simulate how much too good the best of m noisy scores is.
5. `empirical_rademacher`: how well a set of models fits random signs.
6. `nested_cv_score`: choose a ridge penalty inside each outer training set and
   score the choice on the outer fold.

Three more functions are provided, implemented: `k_fold_indices` and
`cross_val_score` from the chapter "Cross-validation and the leave-one-out
shortcut", and `finite_class_bound`, the formula from this chapter.
Wherever `fit` and `predict` appear as arguments, they follow the pattern
`model = fit(X_train, y_train)` and `y_pred = predict(model, X_test)`.
Shapes: X is (n, d), one row per house; y is (n,).
"""

import numpy as np


# Provided: you wrote this in the chapter "Cross-validation and the
# leave-one-out shortcut".
def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Partition {0, ..., n - 1} into k sorted folds of nearly equal size."""
    order = rng.permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


# Provided: also from the cross-validation chapter.
def cross_val_score(fit, predict, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Held-out mean squared error of each of k folds, on folds from seed 0."""
    n = X.shape[0]
    folds = k_fold_indices(n, k, np.random.default_rng(0))
    scores = np.zeros(k)
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        model = fit(X[train], y[train])
        scores[f] = np.mean((predict(model, X[test]) - y[test]) ** 2)
    return scores


# Provided: the finite-class bound from this chapter. It needs losses in [0, 1].
def finite_class_bound(M: int, n: int, delta: float) -> float:
    """With probability at least 1 - delta, every one of M models has a training
    average within this distance of its true average, for n i.i.d. examples."""
    return float(np.sqrt(np.log(2 * M / delta) / (2 * n)))


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
    raise NotImplementedError("validation_split")


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
    raise NotImplementedError("learning_curve")


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
    raise NotImplementedError("diagnose_curve")


def best_of_m_gap(
    true_error: float, noise_sd: float, m: int, trials: int, rng: np.random.Generator
) -> float:
    """Simulate how much too good the best of m equal models looks.

    All m models have the same true error. Each one's estimated error is
    `true_error` plus independent normal noise of standard deviation
    `noise_sd`. In each trial you pick the model with the lowest estimate.

    Args:
        true_error: The true error shared by all m models.
        noise_sd: Standard deviation of each model's estimation noise.
        m: Number of models tried.
        trials: Number of simulated repeats.
        rng: Source of randomness.

    Returns:
        The average over trials of `true_error` minus the lowest estimate.
    """
    raise NotImplementedError("best_of_m_gap")


def empirical_rademacher(
    losses_matrix: np.ndarray, rng: np.random.Generator, num_draws: int = 2000
) -> float:
    """Estimate how well a finite set of models fits random signs.

    `losses_matrix[j, i]` is model j's output on example i. Draw `num_draws`
    random sign vectors sigma in {-1, +1}^n. For each draw, score every model
    by (1 / n) * sigma . f_j and keep the best score. Return the average of
    those best scores.

    Args:
        losses_matrix: Array of shape (num_models, n).
        rng: Source of randomness for the signs.
        num_draws: Number of random sign vectors.

    Returns:
        The Monte Carlo estimate, one float.
    """
    raise NotImplementedError("empirical_rademacher")


def nested_cv_score(X: np.ndarray, y: np.ndarray, lambdas, outer_k: int, inner_k: int) -> float:
    """Score the recipe "pick the ridge penalty by cross-validation".

    The outer folds come from `k_fold_indices(n, outer_k, np.random.default_rng(0))`,
    the same folds `cross_val_score` uses. For each outer fold:
      1. On the outer training rows only, score every penalty with
         `cross_val_score` and `inner_k` folds, and pick the one with the
         lowest mean.
      2. Fit ridge with that penalty on all outer training rows.
      3. Measure the mean squared error on the outer test fold.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lambdas: Sequence of nonnegative ridge penalties.
        outer_k: Number of outer folds.
        inner_k: Number of inner folds.

    Returns:
        The mean of the outer fold errors, one float.
    """
    raise NotImplementedError("nested_cv_score")


def demo_learning_curves() -> None:
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


def demo_winners_curse() -> None:
    rng = np.random.default_rng(0)
    for m in (1, 2, 5, 20, 100):
        gap = best_of_m_gap(true_error=1.0, noise_sd=0.1, m=m, trials=20000, rng=rng)
        print(f"best of {m:>3} equal models looks {gap:.3f} better than the truth")

    one = rng.choice(np.array([-1.0, 1.0]), size=(1, 40))
    many = rng.choice(np.array([-1.0, 1.0]), size=(300, 40))
    print(f"fits noise: 1 model {empirical_rademacher(one, rng, 4000):.3f}, "
          f"300 models {empirical_rademacher(many, rng, 4000):.3f}")

    # The sixty houses from the cross-validation chapter.
    r = np.random.default_rng(7)
    n, d = 60, 30
    X = r.normal(size=(n, 5)) @ r.normal(size=(5, d))
    X = X + 0.5 * r.normal(size=(n, d))
    y = X @ (0.3 * r.normal(size=d)) + r.normal(size=n)
    lambdas = np.logspace(-2, 3, 20)
    ridge_fit = lambda Xt, yt, lam: np.linalg.solve(Xt.T @ Xt + lam * np.eye(d), Xt.T @ yt)
    best_naive = min(
        cross_val_score(lambda a, b, lam=lam: ridge_fit(a, b, lam), lambda w, Xs: Xs @ w, X, y, 5).mean()
        for lam in lambdas
    )
    print(f"best of 20 penalties, scored on the data that chose it: {best_naive:.3f}")
    print(f"nested cross-validation, the honest score:              {nested_cv_score(X, y, lambdas, 5, 4):.3f}")


if __name__ == "__main__":
    demo_learning_curves()
    print()
    demo_winners_curse()
