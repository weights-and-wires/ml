"""Reference solution. Gaussian discriminant analysis: one Gaussian cloud per class, one shared shape.

Fill in `fit_gda`, `gaussian_log_density`, and `gda_log_posterior`.
`log_sum_exp` and `log_posterior_from_log_density` are given, because you
wrote them in the chapter on Bayes' rule.

Labels `y` are integers `0, ..., K - 1`.
"""

import numpy as np


# Given: you wrote this in the chapter "Classify by asking who made this".
def log_sum_exp(a, axis):
    """Return log(sum(exp(a))) along `axis`, without overflow or underflow."""
    a = np.asarray(a, dtype=float)
    top = np.max(a, axis=axis, keepdims=True)
    total = np.sum(np.exp(a - top), axis=axis, keepdims=True)
    return np.squeeze(top + np.log(total), axis=axis)


# Given: you wrote this in the chapter "Classify by asking who made this".
def log_posterior_from_log_density(log_density, priors):
    """Turn `(n, K)` log densities and `(K,)` priors into `(n, K)` log posteriors."""
    joint = log_density + np.log(priors)
    return joint - log_sum_exp(joint, axis=1)[:, None]


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


if __name__ == "__main__":
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
