"""Checks for the averaging-models exercise."""

import numpy as np

from mlcheck import almost, check, fail, has, shape


def formula(rho, sigma2, B):
    return rho * sigma2 + (1 - rho) * sigma2 / B


@check("bagging_variance matches its extremes and the worked example")
def test_bagging_variance_extremes(sub):
    got = has(sub, "bagging_variance")
    almost(got(0.0, 2.0, 10), 0.2, tol=1e-10, label="rho=0, sigma2=2, B=10")
    almost(got(1.0, 2.0, 10), 2.0, tol=1e-10, label="rho=1, sigma2=2, B=10")
    almost(got(0.4, 3.0, 1), 3.0, tol=1e-10, label="B=1 returns one model's variance")
    almost(got(0.5, 4.0, 2), 3.0, tol=1e-10, label="rho=0.5, sigma2=4, B=2")


@check("bagging_variance keeps the shared part whole as B grows")
def test_bagging_variance_floor(sub):
    got = has(sub, "bagging_variance")
    almost(got(0.3, 1.0, 50), 0.314, tol=1e-10, label="rho=0.3, sigma2=1, B=50")
    almost(got(0.3, 1.0, 5000), 0.30014, tol=1e-10, label="rho=0.3, sigma2=1, B=5000")
    almost(got(0.8, 1.0, 10**7), 0.8, tol=1e-6, label="B huge approaches rho * sigma2")


@check("correlated_estimators returns an (n_draws, B) array and repeats under the same seed")
def test_estimators_shape(sub):
    gen = has(sub, "correlated_estimators")
    a = gen(0.4, 2.0, 7, 30, np.random.default_rng(0))
    b = gen(0.4, 2.0, 7, 30, np.random.default_rng(0))
    shape(a, (30, 7), label="correlated_estimators output")
    if not np.allclose(a, b):
        fail("two calls with the same seed gave different arrays, so the draws are not coming from rng")
    if np.allclose(a[:, 0], a[:, 1]):
        fail("columns 0 and 1 are identical, but rho=0.4 should leave each column its own private part")


@check("each column of correlated_estimators has variance sigma2 and mean 0")
def test_estimators_variance(sub):
    gen = has(sub, "correlated_estimators")
    rng = np.random.default_rng(1)
    sigma2 = 2.5
    for rho in (0.0, 0.3, 0.8):
        draws = gen(rho, sigma2, 4, 40000, rng)
        for b in range(4):
            almost(float(draws[:, b].var()), sigma2, tol=0.06 * sigma2, label=f"variance of column {b} at rho={rho}")
        almost(float(draws.mean()), 0.0, tol=0.03, label=f"mean at rho={rho}")


@check("each pair of columns in correlated_estimators has correlation rho")
def test_estimators_correlation(sub):
    gen = has(sub, "correlated_estimators")
    rng = np.random.default_rng(2)
    for rho in (0.0, 0.3, 0.8, 1.0):
        draws = gen(rho, 1.7, 4, 40000, rng)
        corr = np.corrcoef(draws, rowvar=False)
        off = corr[~np.eye(4, dtype=bool)]
        for value in off:
            almost(float(value), rho, tol=0.03, label=f"pairwise correlation at rho={rho}")


@check("the variance of the average of correlated_estimators matches bagging_variance")
def test_estimators_average(sub):
    gen = has(sub, "correlated_estimators")
    got = has(sub, "bagging_variance")
    rng = np.random.default_rng(3)
    sigma2 = 4.0
    for rho, B in [(0.5, 2), (0.3, 10), (0.8, 25), (0.0, 8)]:
        draws = gen(rho, sigma2, B, 60000, rng)
        empirical = float(draws.mean(axis=1).var())
        almost(empirical, got(rho, sigma2, B), tol=0.04 * sigma2, label=f"rho={rho}, B={B}: simulated vs formula")
        almost(empirical, formula(rho, sigma2, B), tol=0.04 * sigma2, label=f"rho={rho}, B={B}: simulated vs hand formula")


@check("trees_needed returns the smallest B within the tolerance of the floor")
def test_trees_needed(sub):
    need = has(sub, "trees_needed")
    if need(0.3, 1.0, 0.05) != 47:
        fail(f"trees_needed(0.3, 1.0, 0.05) should be 47, got {need(0.3, 1.0, 0.05)}")
    for rho, sigma2, tol in [(0.3, 1.0, 0.07), (0.6, 2.0, 0.013), (0.9, 5.0, 0.021), (0.15, 0.5, 0.11)]:
        B = need(rho, sigma2, tol)
        floor = rho * sigma2
        if not isinstance(B, (int, np.integer)):
            fail(f"trees_needed must return an integer, got {B!r}")
        if formula(rho, sigma2, B) > (1 + tol) * floor + 1e-12:
            fail(f"B={B} leaves variance {formula(rho, sigma2, B):.6f}, above {(1 + tol) * floor:.6f} for rho={rho}, tol={tol}")
        if B > 1 and formula(rho, sigma2, B - 1) <= (1 + tol) * floor - 1e-12:
            fail(f"B={B} is not the smallest: B={B - 1} already reaches the tolerance for rho={rho}, tol={tol}")


@check("trees_needed returns 1 for a huge tolerance and rejects rho = 0")
def test_trees_needed_edges(sub):
    need = has(sub, "trees_needed")
    if need(0.5, 1.0, 100.0) != 1:
        fail(f"a huge tolerance needs only 1 model, got {need(0.5, 1.0, 100.0)}")
    try:
        need(0.0, 1.0, 0.05)
    except ValueError:
        return
    fail("trees_needed(0.0, ...) should raise ValueError, because the floor is 0")
