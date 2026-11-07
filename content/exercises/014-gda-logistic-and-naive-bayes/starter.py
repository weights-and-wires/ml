"""From Gaussian classes to logistic regression and naive Bayes.

You write six functions, in this order:

1. `gda_to_logistic`: the logistic weights and bias that two Gaussians with one shared covariance imply.
2. `log_odds`: log p(class 1 | x) minus log p(class 0 | x), from the GDA posterior.
3. `decision_boundary_1d`: the single x where two one-dimensional classes are equally probable.
4. `fit_naive_bayes_gaussian`: one Gaussian per class and feature, with a variance floor.
5. `fit_naive_bayes_multinomial`: smoothed word probabilities for count data.
6. `multinomial_nb_log_posterior`: score a document by summing count times log probability.

The functions from the chapter "Bayes' rule and Gaussian classes", `sigmoid`,
and `gaussian_nb_log_posterior` are given.
Labels `y` are integers `0, ..., K - 1`.
"""

import numpy as np


# Given: you wrote this in the chapter "Bayes' rule and Gaussian classes".
def log_sum_exp(a, axis):
    """Return log(sum(exp(a))) along `axis`, without overflow or underflow."""
    a = np.asarray(a, dtype=float)
    top = np.max(a, axis=axis, keepdims=True)
    total = np.sum(np.exp(a - top), axis=axis, keepdims=True)
    return np.squeeze(top + np.log(total), axis=axis)


# Given: you wrote this in the chapter "Bayes' rule and Gaussian classes".
def log_posterior_from_log_density(log_density, priors):
    """Turn `(n, K)` log densities and `(K,)` priors into `(n, K)` log posteriors."""
    joint = log_density + np.log(priors)
    return joint - log_sum_exp(joint, axis=1)[:, None]


# Given: you wrote this in the chapter "Bayes' rule and Gaussian classes".
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


# Given: you wrote this in the chapter "Bayes' rule and Gaussian classes".
def gaussian_log_density(X, mean, cov):
    """Log of the multivariate Gaussian density at every row of `X`, shape `(n,)`."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    distance = np.sum(z**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det - 0.5 * distance


# Given: you wrote this in the chapter "Bayes' rule and Gaussian classes".
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


# Given: you wrote this in the chapter "Bayes' rule and Gaussian classes".
def normal_log_density_1d(x, mean, var):
    """Log of the one-dimensional Gaussian density at `x`."""
    x = np.asarray(x, dtype=float)
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mean) ** 2 / (2.0 * var)


# Given: read how a Gaussian naive Bayes model scores points.
def gaussian_nb_log_posterior(X, means, variances, priors):
    """Log posterior `(n, K)`: per-feature log densities summed, then Bayes' rule."""
    K = means.shape[0]
    log_density = np.stack(
        [normal_log_density_1d(X, means[k], variances[k]).sum(axis=1) for k in range(K)],
        axis=1,
    )
    return log_posterior_from_log_density(log_density, priors)


def gda_to_logistic(means, cov, priors):
    """Return the weights `w` `(d,)` and bias `b` (a float) of the equivalent logistic regression.

    `means` has shape `(2, d)`, `cov` is `(d, d)`, and `priors` has shape `(2,)`.
    For a point `x`, `sigmoid(x @ w + b)` is the GDA probability of class 1.
    """
    raise NotImplementedError("w from the shape-weighted mean difference, b from the midpoint and priors")


def log_odds(X, means, cov, priors):
    """Return log p(class 1 | x) - log p(class 0 | x) for every row of `X`, shape `(n,)`.

    Build it from `gda_log_posterior`. Two classes only.
    """
    raise NotImplementedError("subtract two columns of the log posterior")


def decision_boundary_1d(means, var, priors):
    """Return the single x where the two classes are equally probable.

    `means` holds the two class means (shape `(2,)` or `(2, 1)`), `var` is the
    shared variance (a float), and `priors` has shape `(2,)`. Returns a float.
    """
    raise NotImplementedError("solve w * x + b = 0 in one dimension")


def fit_naive_bayes_gaussian(X, y, var_floor=1e-9):
    """Fit one Gaussian per class and feature.

    `X` has shape `(n, d)` and `y` holds integer labels of shape `(n,)`, with
    `K = y.max() + 1` classes. Returns `means` `(K, d)`, `variances` `(K, d)`,
    and `priors` `(K,)`. Each variance is the average squared gap from that
    class's own mean, over that class's rows, and never smaller than
    `var_floor`.
    """
    raise NotImplementedError("per class: mean, variance with a floor, and prior")


def fit_naive_bayes_multinomial(X, y, alpha=1.0):
    """Fit smoothed word probabilities for count data.

    `X` holds word counts of shape `(n, V)`. Returns `log_theta` `(K, V)`, the
    log of each class's word probabilities, and `log_priors` `(K,)`. Add
    `alpha` to every word count of a class, then divide by the smoothed total, so
    each row of `np.exp(log_theta)` adds to 1.
    """
    raise NotImplementedError("sum the class rows, add alpha, normalise, take the log")


def multinomial_nb_log_posterior(X, log_theta, log_priors):
    """Log posterior `(n, K)` for rows of word counts.

    `X` is `(n, V)` counts, `log_theta` is `(K, V)`, and `log_priors` is
    `(K,)`. A class's score for a document is the sum over words of count times
    log probability, plus the log prior. Subtract the log-sum-exp across classes.
    """
    raise NotImplementedError("counts times log theta, plus log prior, then normalise")


def demo_gda_as_logistic() -> None:
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


def demo_naive_bayes() -> None:
    # The toy spam set from the chapter. Words: free, money, win, meeting, report, lunch.
    docs = np.array(
        [[2, 1, 1, 0, 0, 0], [1, 2, 0, 0, 1, 0], [0, 0, 0, 2, 1, 1], [1, 0, 0, 1, 2, 0]]
    )
    labels = np.array([1, 1, 0, 0])  # 1 is spam, 0 is ham
    log_theta, log_priors = fit_naive_bayes_multinomial(docs, labels, alpha=1.0)
    print("ham word probabilities:", np.exp(log_theta[0]).round(3))
    doc = np.array([[2, 1, 0, 1, 0, 0]])
    post = np.exp(multinomial_nb_log_posterior(doc, log_theta, log_priors))
    print("posterior [ham, spam] for 'free free money meeting':", post.round(3))

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0, 0], 1.0, size=(50, 3)), rng.normal([2, 1, 0], 1.0, size=(50, 3))])
    y = np.repeat([0, 1], 50)
    means, variances, priors = fit_naive_bayes_gaussian(X, y)
    acc = (gaussian_nb_log_posterior(X, means, variances, priors).argmax(axis=1) == y).mean()
    print("per-class variances:\n", variances.round(2))
    print("training accuracy:", acc)


if __name__ == "__main__":
    demo_gda_as_logistic()
    print()
    demo_naive_bayes()
