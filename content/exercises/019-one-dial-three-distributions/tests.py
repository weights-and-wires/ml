"""Checks for the one-dial exercise."""

from math import factorial, pi, sqrt

import numpy as np

from mlcheck import check, close, fail, shape

ETAS = [-1.5, -0.3, 0.0, 0.7, 2.0]


def softplus_ref(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


@check("log_partition gives eta^2/2, softplus, and e^eta for the three families")
def test_log_partition(sub):
    eta = np.array([-1.0, 0.0, 1.0, 2.0])
    close(sub.log_partition("gaussian", eta), 0.5 * eta ** 2, atol=1e-10, label="gaussian A")
    close(sub.log_partition("bernoulli", eta), softplus_ref(eta), atol=1e-10, label="bernoulli A")
    close(sub.log_partition("poisson", eta), np.exp(eta), atol=1e-8, label="poisson A")
    out = sub.log_partition("bernoulli", np.array([-800.0, 800.0]))
    if not np.all(np.isfinite(out)):
        fail(f"bernoulli A at eta = -800 and 800 returned {out}, expected finite values")
    close(out, np.array([0.0, 800.0]), atol=1e-9, label="bernoulli A at large |eta|")


@check("log_partition is the log of the total, so the probabilities add up to 1")
def test_normaliser(sub):
    for eta in (-1.2, 0.0, 0.9):
        e = np.array([eta])
        # Bernoulli: h(y) = 1 for y in {0, 1}.
        y = np.array([0.0, 1.0])
        total = np.sum(np.exp(eta * y - sub.log_partition("bernoulli", e)))
        if abs(total - 1.0) > 1e-9:
            fail(f"bernoulli probabilities at eta={eta} add up to {total:.6f}, expected 1")
        # Poisson: h(y) = 1 / y!.
        y = np.arange(0, 80)
        h = np.array([1.0 / factorial(int(k)) if k < 25 else 0.0 for k in y])
        total = np.sum(h * np.exp(eta * y - sub.log_partition("poisson", e)))
        if abs(total - 1.0) > 1e-9:
            fail(f"poisson probabilities at eta={eta} add up to {total:.6f}, expected 1")
        # Gaussian: h(y) = exp(-y^2 / 2) / sqrt(2 pi), integrated on a grid.
        grid = np.linspace(-12.0, 12.0, 4801)
        dens = np.exp(-grid ** 2 / 2) / sqrt(2 * pi) * np.exp(eta * grid - sub.log_partition("gaussian", e))
        total = np.sum(dens) * (grid[1] - grid[0])
        if abs(total - 1.0) > 1e-6:
            fail(f"gaussian density at eta={eta} integrates to {total:.6f}, expected 1")


@check("mean_from_natural matches the slope of log_partition")
def test_mean_is_slope(sub):
    eps = 1e-6
    for family in ("gaussian", "bernoulli", "poisson"):
        for e0 in ETAS:
            numeric = (sub.log_partition(family, np.array([e0 + eps]))[0]
                       - sub.log_partition(family, np.array([e0 - eps]))[0]) / (2 * eps)
            got = float(sub.mean_from_natural(family, np.array([e0]))[0])
            if abs(got - numeric) > 1e-4 * max(1.0, abs(numeric)):
                fail(f"{family} mean at eta={e0}: got {got:.6g}, the slope of log_partition is {numeric:.6g}")
    shape(sub.mean_from_natural("poisson", np.zeros((2, 3))), (2, 3), label="mean output")


@check("variance_from_natural matches the bend of log_partition")
def test_variance_is_bend(sub):
    eps = 1e-3
    for family in ("gaussian", "bernoulli", "poisson"):
        for e0 in ETAS:
            def A(v):
                return sub.log_partition(family, np.array([v]))[0]
            numeric = (A(e0 + eps) - 2 * A(e0) + A(e0 - eps)) / eps ** 2
            got = float(sub.variance_from_natural(family, np.array([e0]))[0])
            if abs(got - numeric) > 5e-3 * max(1.0, abs(numeric)):
                fail(f"{family} variance at eta={e0}: got {got:.6g}, the bend of log_partition is {numeric:.6g}")


@check("the mean is the average y under the Poisson and Bernoulli probabilities")
def test_mean_is_average(sub):
    for eta in (-0.7, 0.0, 1.1):
        e = np.array([eta])
        y = np.arange(0, 25)
        h = np.array([1.0 / factorial(int(k)) for k in y])
        p = h * np.exp(eta * y - sub.log_partition("poisson", e))
        mean = np.sum(y * p)
        var = np.sum((y - mean) ** 2 * p)
        close(sub.mean_from_natural("poisson", e)[0], mean, atol=1e-8, label=f"poisson mean at eta={eta}")
        close(sub.variance_from_natural("poisson", e)[0], var, atol=1e-8, label=f"poisson variance at eta={eta}")
        pb = np.exp(eta * np.array([0.0, 1.0]) - sub.log_partition("bernoulli", e))
        close(sub.mean_from_natural("bernoulli", e)[0], pb[1], atol=1e-10, label=f"bernoulli mean at eta={eta}")
        close(sub.variance_from_natural("bernoulli", e)[0], pb[1] * pb[0], atol=1e-10,
              label=f"bernoulli variance at eta={eta}")


@check("the Poisson variance equals its mean, and the Gaussian variance is 1")
def test_variance_ties(sub):
    eta = np.array([-2.0, -0.5, 0.0, 0.5, 1.5])
    shape(sub.variance_from_natural("gaussian", eta), (5,), label="gaussian variance output (return an array, not the bare number 1)")
    close(sub.variance_from_natural("poisson", eta), sub.mean_from_natural("poisson", eta),
          atol=1e-10, label="Poisson variance should equal Poisson mean")
    close(sub.variance_from_natural("gaussian", eta), np.ones(5), atol=1e-12, label="gaussian variance")
    close(sub.mean_from_natural("bernoulli", np.array([0.0]))[0], 0.5, atol=1e-12, label="bernoulli mean at 0")


@check("the Bernoulli mean and variance stay finite at extreme dial settings")
def test_stability(sub):
    eta = np.array([-800.0, 800.0])
    try:
        with np.errstate(over="raise"):
            sub.mean_from_natural("bernoulli", eta)
            sub.variance_from_natural("bernoulli", eta)
    except FloatingPointError:
        fail("the Bernoulli mean or variance overflowed at eta = -800 or 800; write the sigmoid so that np.exp never sees a large positive number")
    m = sub.mean_from_natural("bernoulli", eta)
    v = sub.variance_from_natural("bernoulli", eta)
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(v))):
        fail(f"bernoulli mean {m} and variance {v} at eta = -800, 800 should be finite")
    close(m, np.array([0.0, 1.0]), atol=1e-12, label="bernoulli mean at extremes")
    close(v, np.array([0.0, 0.0]), atol=1e-12, label="bernoulli variance at extremes")
    try:
        sub.log_partition("cauchy", np.array([0.0]))
    except ValueError:
        return
    fail("log_partition should raise ValueError for an unknown family")
