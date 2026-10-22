"""Reference solution. Two Gaussians with one shared covariance are a logistic regression.

Fill in `gda_to_logistic`, `log_odds`, and `decision_boundary_1d`. The
functions from the two earlier chapters, and `sigmoid`, are given.
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


# Given: you wrote this in the chapter "Fit a Gaussian cloud to each class".
def fit_gda(X, y):
    """Return class means `(K, d)`, a shared covariance `(d, d)`, and priors `(K,)`."""
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


# Given: you wrote this in the chapter "Fit a Gaussian cloud to each class".
def gaussian_log_density(X, mean, cov):
    """Log of the multivariate Gaussian density at every row of `X`, shape `(n,)`."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    distance = np.sum(z**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det - 0.5 * distance


# Given: you wrote this in the chapter "Fit a Gaussian cloud to each class".
def gda_log_posterior(X, means, cov, priors):
    """Log posterior of every class for every row of `X`, shape `(n, K)`."""
    log_density = np.stack(
        [gaussian_log_density(X, mean, cov) for mean in means], axis=1
    )
    return log_posterior_from_log_density(log_density, priors)


# Given: the sigmoid from the logistic regression chapters.
def sigmoid(z):
    """Return 1 / (1 + exp(-z)) without overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def gda_to_logistic(means, cov, priors):
    """Return the weights `w` `(d,)` and bias `b` (a float) of the equivalent logistic regression.

    `means` has shape `(2, d)`, `cov` is `(d, d)`, and `priors` has shape `(2,)`.
    For a point `x`, `sigmoid(x @ w + b)` is the GDA probability of class 1.
    """
    w = np.linalg.solve(cov, means[1] - means[0])
    b = np.log(priors[1] / priors[0]) - 0.5 * (means[0] + means[1]) @ w
    return w, float(b)


def log_odds(X, means, cov, priors):
    """Return log p(class 1 | x) - log p(class 0 | x) for every row of `X`, shape `(n,)`.

    Build it from `gda_log_posterior`. Two classes only.
    """
    log_post = gda_log_posterior(X, means, cov, priors)
    return log_post[:, 1] - log_post[:, 0]


def decision_boundary_1d(means, var, priors):
    """Return the single x where the two classes are equally probable.

    `means` holds the two class means (shape `(2,)` or `(2, 1)`), `var` is the
    shared variance (a float), and `priors` has shape `(2,)`. Returns a float.
    """
    mu0, mu1 = np.ravel(means)
    w = (mu1 - mu0) / var
    b = np.log(priors[1] / priors[0]) - w * (mu0 + mu1) / 2.0
    return float(-b / w)


if __name__ == "__main__":
    X = np.array([[1.0], [3.0], [5.0], [7.0]])
    y = np.array([0, 0, 1, 1])
    means, cov, priors = fit_gda(X, y)
    w, b = gda_to_logistic(means, cov, priors)
    print("w:", w, "b:", b)
    print("probability at x = 3:", sigmoid(np.array([3.0]) @ w + b))
    print("boundary:", decision_boundary_1d(means, float(cov[0, 0]), priors))

    rng = np.random.default_rng(0)
    shape = np.array([[1.5, 1.2], [1.2, 1.5]])
    A = rng.multivariate_normal([0, 0], shape, size=50)
    B = rng.multivariate_normal([3, 0], shape, size=50)
    X2, y2 = np.vstack([A, B]), np.repeat([0, 1], 50)
    means2, cov2, priors2 = fit_gda(X2, y2)
    w2, b2 = gda_to_logistic(means2, cov2, priors2)
    p_gda = np.exp(gda_log_posterior(X2, means2, cov2, priors2))[:, 1]
    p_logistic = sigmoid(X2 @ w2 + b2)
    print("largest gap between GDA and sigmoid(w.x + b):", np.abs(p_gda - p_logistic).max())
    print("mean difference:", means2[1] - means2[0], "weights:", w2.round(2))
