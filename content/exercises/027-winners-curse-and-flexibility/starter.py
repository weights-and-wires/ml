"""Measure the winner's curse, and score a choice honestly.

You write three functions, in this order:

1. `best_of_m_gap`: simulate how much too good the best of m noisy scores is.
2. `empirical_rademacher`: how well a set of models fits random signs.
3. `nested_cv_score`: choose a ridge penalty inside each outer training set and
   score the choice on the outer fold.

Three more functions are provided, implemented: `k_fold_indices` and
`cross_val_score` from the chapter on cross-validation, and
`finite_class_bound`, the formula from this chapter.
Shapes: X is (n, d), one row per house; y is (n,).
"""

import numpy as np


# Provided: you wrote this in the chapter "Why training error lies, and how
# k-fold cross-validation fixes it".
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


if __name__ == "__main__":
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
