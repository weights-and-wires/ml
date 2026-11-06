"""One dial, three distributions: mean and variance from the log-partition.

Three families share one shape. Each is described by a single function of a
dial `eta`, the log-partition A(eta):

    p(y | eta) = h(y) * exp(eta * y - A(eta))

The slope of A is the mean and the bend of A is the variance. You write three
functions, in this order:

1. `log_partition`: A(eta) for each family.
2. `mean_from_natural`: the mean, which is the slope A'(eta).
3. `variance_from_natural`: the variance, which is the bend A''(eta).

The families are `"gaussian"` (variance 1), `"bernoulli"`, and `"poisson"`.
"""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    """Compute the log-partition function A(eta) for one family.

    Args:
        family: One of `"gaussian"` (variance 1), `"bernoulli"`, or `"poisson"`.
        eta: Dial values, any shape.

    Returns:
        A(eta), same shape as `eta`. The Bernoulli case must stay finite for
        `eta` as large in size as 800, so avoid `np.log(1 + np.exp(eta))`.
        Raise `ValueError` for an unknown family.
    """
    raise NotImplementedError("log_partition")


def mean_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    """Compute the mean A'(eta) implied by the dial.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Dial values, any shape.

    Returns:
        The mean, same shape as `eta`. Stay finite for very large or very
        small `eta`. Raise `ValueError` for an unknown family.
    """
    raise NotImplementedError("mean_from_natural")


def variance_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    """Compute the variance A''(eta) implied by the dial.

    Args:
        family: One of `"gaussian"`, `"bernoulli"`, or `"poisson"`.
        eta: Dial values, any shape.

    Returns:
        The variance, same shape as `eta`. Raise `ValueError` for an unknown
        family.
    """
    raise NotImplementedError("variance_from_natural")


if __name__ == "__main__":
    eta = np.array([-2.0, 0.0, 2.0])
    for family in FAMILIES:
        print(family)
        print("  A(eta)   :", np.round(log_partition(family, eta), 4))
        print("  mean     :", np.round(mean_from_natural(family, eta), 4))
        print("  variance :", np.round(variance_from_natural(family, eta), 4))
    h = 1e-3
    slope = (log_partition("poisson", 0.0 + h) - log_partition("poisson", 0.0 - h)) / (2 * h)
    print("Poisson slope of A at 0:", round(float(slope), 4), "(the mean is 1)")
