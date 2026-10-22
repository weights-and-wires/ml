"""Reference solution for the one-dial exercise."""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return 0.5 * eta ** 2
    if family == "bernoulli":
        return np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def mean_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return eta.copy()
    if family == "bernoulli":
        return _sigmoid(eta)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def variance_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return np.ones_like(eta)
    if family == "bernoulli":
        p = _sigmoid(eta)
        return p * (1.0 - p)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


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
