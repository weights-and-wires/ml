"""Bayes' rule and Gaussian classes, built from six small functions.

You write six functions, in this order:

1. `normal_log_density_1d`: the log of a one-dimensional Gaussian density.
2. `log_sum_exp`: a log of a sum of exponentials that never overflows or underflows.
3. `log_posterior_from_log_density`: Bayes' rule, in logs.
4. `fit_gda`: class means, one shared covariance, and priors, by averaging.
5. `gaussian_log_density`: the multivariate Gaussian log density, with a Cholesky factor.
6. `gda_log_posterior`: one log density per class, then Bayes' rule.

`predict_class` is given. Every function works in logs, so a product of tiny
numbers never underflows to zero. Labels `y` are integers `0, ..., K - 1`.
"""

import numpy as np


def normal_log_density_1d(x, mean, var):
    """Log of the one-dimensional Gaussian density at `x`.

    `x` is an array of any shape. `mean` and `var` are scalars, or arrays that
    broadcast against `x`, and `var` is the variance (the square of the
    standard deviation). Returns an array with the shape of `x`. A density is
    a height, not a probability, so the result can be above zero.
    """
    x = np.asarray(x, dtype=float)
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mean) ** 2 / (2.0 * var)


def log_sum_exp(a, axis):
    """Return log(sum(exp(a))) along `axis`, without overflow or underflow.

    `a` is an array. Returns an array with `axis` removed, so a `(n, K)` input
    with `axis=1` gives shape `(n,)`. Subtract the maximum along `axis` before
    you exponentiate, and add it back after the log.
    """
    a = np.asarray(a, dtype=float)
    top = np.max(a, axis=axis, keepdims=True)
    total = np.sum(np.exp(a - top), axis=axis, keepdims=True)
    return np.squeeze(top + np.log(total), axis=axis)


def log_posterior_from_log_density(log_density, priors):
    """Apply Bayes' rule in logs.

    `log_density` has shape `(n, K)`: the log of p(x | class k) for each of `n`
    points and `K` classes. `priors` has shape `(K,)` and adds to 1. Returns
    an `(n, K)` array of log p(class k | x), so that `np.exp(...)` sums to 1
    along each row.
    """
    joint = log_density + np.log(priors)
    return joint - log_sum_exp(joint, axis=1)[:, None]


# Given: the winner is the class with the largest posterior.
def predict_class(log_posterior):
    """Return the most probable class for each row of an `(n, K)` array."""
    return np.argmax(log_posterior, axis=1)


def fit_gda(X, y):
    """Fit class means, one shared covariance, and class priors.

    `X` has shape `(n, d)` and `y` holds integer labels of shape `(n,)`, with
    `K = y.max() + 1` classes. Returns `means` of shape `(K, d)`, `cov` of
    shape `(d, d)`, and `priors` of shape `(K,)`. The covariance averages the
    outer products of each row's gap from its own class mean, and divides by
    `n`, the total number of rows.
    """
    n, d = X.shape
    K = int(y.max()) + 1
    means = np.zeros((K, d))
    priors = np.zeros(K)
    cov = np.zeros((d, d))
    for k in range(K):
        rows = X[y == k]
        priors[k] = len(rows) / n
        means[k] = rows.mean(axis=0)
        diff = rows - means[k]
        cov += diff.T @ diff
    return means, cov / n, priors


def gaussian_log_density(X, mean, cov):
    """Log of the multivariate Gaussian density at every row of `X`.

    `X` has shape `(n, d)`, `mean` has shape `(d,)`, and `cov` is a `(d, d)`
    symmetric positive definite matrix. Returns shape `(n,)`. Factor `cov` with
    a Cholesky decomposition instead of inverting it.
    """
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    distance = np.sum(z**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det - 0.5 * distance


def gda_log_posterior(X, means, cov, priors):
    """Log posterior of every class for every row of `X`.

    Returns shape `(n, K)`, and `np.exp` of it adds to 1 along each row. Use
    `gaussian_log_density` for each class, then Bayes' rule.
    """
    log_density = np.stack(
        [gaussian_log_density(X, mean, cov) for mean in means], axis=1
    )
    return log_posterior_from_log_density(log_density, priors)


def demo_bayes_rule() -> None:
    # Warehouse 0 ships light parcels, warehouse 1 heavy ones, and 90% of
    # parcels come from warehouse 0. A parcel weighs 3 kg.
    x = np.array([3.0])
    log_density = np.stack(
        [normal_log_density_1d(x, 1.5, 1.0), normal_log_density_1d(x, 4.0, 1.0)],
        axis=1,
    )
    print("densities at 3 kg:", np.exp(log_density).round(4))
    for priors in ([0.5, 0.5], [0.9, 0.1]):
        post = np.exp(log_posterior_from_log_density(log_density, np.array(priors)))
        print(f"priors {priors}: posterior {post.round(4)}, class {predict_class(np.log(post))}")


def demo_gda() -> None:
    # The four-point example from the chapter: classes at 1, 3 and 5, 7.
    X = np.array([[1.0], [3.0], [5.0], [7.0]])
    y = np.array([0, 0, 1, 1])
    means, cov, priors = fit_gda(X, y)
    print("means:", means.ravel(), "cov:", cov.ravel(), "priors:", priors)
    post = np.exp(gda_log_posterior(np.array([[3.0]]), means, cov, priors))
    print("posterior at x = 3:", post.round(4))

    # Two tilted clouds in 2-D.
    rng = np.random.default_rng(0)
    shape = np.array([[2.0, 1.2], [1.2, 1.0]])
    A = rng.multivariate_normal([0, 0], shape, size=100)
    B = rng.multivariate_normal([3, 1], shape, size=100)
    X2, y2 = np.vstack([A, B]), np.repeat([0, 1], 100)
    means2, cov2, priors2 = fit_gda(X2, y2)
    print("fitted covariance:\n", cov2.round(2))
    acc = (gda_log_posterior(X2, means2, cov2, priors2).argmax(axis=1) == y2).mean()
    print("training accuracy:", acc)


if __name__ == "__main__":
    demo_bayes_rule()
    print()
    demo_gda()
