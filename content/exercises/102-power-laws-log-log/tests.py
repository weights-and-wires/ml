"""Checks for the power-law exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


@check("power_law_eval computes c * x**k, and accepts a list")
def test_eval(sub):
    ev = has(sub, "power_law_eval")
    got = ev([1.0, 2.0, 4.0], 2.0, 2.0)
    close(got, np.array([2.0, 8.0, 32.0]), label="2 * x**2 at x = 1, 2, 4")
    shape(got, (3,), label="output shape")


@check("Doubling x multiplies power_law_eval by 2**k")
def test_doubling(sub):
    ev = has(sub, "power_law_eval")
    x = np.array([1e7, 3.5e8, 2e9])
    k = -0.34
    ratio = ev(2 * x, k, 400.0) / ev(x, k, 400.0)
    close(ratio, np.full(3, 2.0**k), rtol=1e-9, label="loss ratio per doubling")


@check("fit_power_law recovers a planted exponent and coefficient")
def test_fit_clean(sub):
    fit = has(sub, "fit_power_law")
    x = np.geomspace(1.0, 1000.0, 12)
    k, c = fit(x, 3.0 * x**1.7)
    almost(k, 1.7, tol=1e-9, label="exponent")
    if abs(c - np.log(3.0)) < 1e-3:
        fail("fit_power_law returned log(c), the intercept. Return np.exp(intercept) instead.")
    almost(c, 3.0, tol=1e-8, label="coefficient")


@check("fit_power_law reproduces the hand-worked example from the chapter")
def test_fit_hand(sub):
    fit = has(sub, "fit_power_law")
    k, c = fit([1, 2, 4], [1.667, 1.317, 1.040])
    almost(k, -0.3403, tol=2e-4, label="exponent")
    almost(c, 1.6671, tol=2e-3, label="coefficient")
    if not (isinstance(k, float) and isinstance(c, float)):
        fail(f"fit_power_law should return plain floats, got {type(k).__name__} and {type(c).__name__}")


@check("fit_power_law stays close to a planted exponent under noise")
def test_fit_noisy(sub):
    fit = has(sub, "fit_power_law")
    rng = np.random.default_rng(0)
    x = np.geomspace(1e7, 1e9, 30)
    y = 400.0 / x**0.34 * np.exp(rng.normal(0.0, 0.03, x.size))
    k, c = fit(x, y)
    almost(k, -0.34, tol=0.01, label="exponent under noise")
    almost(c, 400.0, tol=40.0, label="coefficient under noise")


@check("extrapolate matches the true law on clean data")
def test_extrapolate_clean(sub):
    ex = has(sub, "extrapolate")
    x = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    y = 1.667 * x**-0.34
    got = ex(x, y, [300.0, 1000.0])
    shape(got, (2,), label="extrapolate output shape")
    close(got, 1.667 * np.array([300.0, 1000.0]) ** -0.34, rtol=1e-6, label="prediction")


@check("extrapolate lands within 10 percent of the truth on noisy data, and the error grows with distance")
def test_extrapolate_noisy(sub):
    ex = has(sub, "extrapolate")
    rng = np.random.default_rng(3)
    x = np.geomspace(1.0, 100.0, 8)
    y = 1.667 * x**-0.34 * np.exp(rng.normal(0.0, 0.03, x.size))
    near = ex(x, y, [100.0])
    far = ex(x, y, [1000.0])[0]
    true_far = 1.667 * 1000.0**-0.34
    err_far = abs(far - true_far) / true_far
    if err_far > 0.10:
        fail(f"prediction at 1000 is {far:.4f}, true value {true_far:.4f}: {err_far:.1%} off, expected under 10%")
    true_near = 1.667 * 100.0**-0.34
    err_near = abs(near[0] - true_near) / true_near
    if not err_near < err_far:
        fail(f"error inside the data ({err_near:.2%}) should be smaller than far outside it ({err_far:.2%})")
