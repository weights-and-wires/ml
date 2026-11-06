"""Bayes' rule for classification, built from three small functions.

Fill in `normal_log_density_1d`, `log_sum_exp`, and
`log_posterior_from_log_density`. `predict_class` is given. Every function
works in logs, so a product of tiny numbers never underflows to zero.
"""

import numpy as np


def normal_log_density_1d(x, mean, var):
    """Log of the one-dimensional Gaussian density at `x`.

    `x` is an array of any shape. `mean` and `var` are scalars, or arrays that
    broadcast against `x`, and `var` is the variance (the square of the
    standard deviation). Returns an array with the shape of `x`. A density is
    a height, not a probability, so the result can be above zero.
    """
    raise NotImplementedError("write the log of the Gaussian density")


def log_sum_exp(a, axis):
    """Return log(sum(exp(a))) along `axis`, without overflow or underflow.

    `a` is an array. Returns an array with `axis` removed, so a `(n, K)` input
    with `axis=1` gives shape `(n,)`. Subtract the maximum along `axis` before
    you exponentiate, and add it back after the log.
    """
    raise NotImplementedError("write the stable log-sum-exp")


def log_posterior_from_log_density(log_density, priors):
    """Apply Bayes' rule in logs.

    `log_density` has shape `(n, K)`: the log of p(x | class k) for each of `n`
    points and `K` classes. `priors` has shape `(K,)` and adds to 1. Returns
    an `(n, K)` array of log p(class k | x), so that `np.exp(...)` sums to 1
    along each row.
    """
    raise NotImplementedError("add the log prior, then subtract the log-sum-exp")


# Given: the winner is the class with the largest posterior.
def predict_class(log_posterior):
    """Return the most probable class for each row of an `(n, K)` array."""
    return np.argmax(log_posterior, axis=1)


if __name__ == "__main__":
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
