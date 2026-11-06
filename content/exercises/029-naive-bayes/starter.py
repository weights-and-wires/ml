"""Naive Bayes: pretend the features are independent once you know the class.

Fill in `fit_naive_bayes_gaussian`, `fit_naive_bayes_multinomial`, and
`multinomial_nb_log_posterior`. The helpers from the first chapter of this
part, and `gaussian_nb_log_posterior`, are given.

Labels `y` are integers `0, ..., K - 1`.
"""

import numpy as np


# Given: you wrote this in the chapter "Classify by asking who made this".
def normal_log_density_1d(x, mean, var):
    """Log of the one-dimensional Gaussian density at `x`."""
    x = np.asarray(x, dtype=float)
    return -0.5 * np.log(2.0 * np.pi * var) - (x - mean) ** 2 / (2.0 * var)


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


# Given: read how a Gaussian naive Bayes model scores points.
def gaussian_nb_log_posterior(X, means, variances, priors):
    """Log posterior `(n, K)`: per-feature log densities summed, then Bayes' rule."""
    K = means.shape[0]
    log_density = np.stack(
        [normal_log_density_1d(X, means[k], variances[k]).sum(axis=1) for k in range(K)],
        axis=1,
    )
    return log_posterior_from_log_density(log_density, priors)


def multinomial_nb_log_posterior(X, log_theta, log_priors):
    """Log posterior `(n, K)` for rows of word counts.

    `X` is `(n, V)` counts, `log_theta` is `(K, V)`, and `log_priors` is
    `(K,)`. A class's score for a document is the sum over words of count times
    log probability, plus the log prior. Subtract the log-sum-exp across classes.
    """
    raise NotImplementedError("counts times log theta, plus log prior, then normalise")


if __name__ == "__main__":
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
