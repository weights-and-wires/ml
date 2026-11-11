"""Checks for the learning-theory exercise."""

import math

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


@check("hoeffding_bound gives 2 exp(-2 n eps^2): 0.2707 at n = 100, eps = 0.1, and caps at 1")
def test_hoeffding(sub):
    # Plausible wrong implementation: exp(-n eps^2), which drops the 2 in front
    # and the 2 in the exponent. It gives 0.3679 here, not 0.2707.
    f = has(sub, "hoeffding_bound")
    almost(f(100, 0.1), 2 * math.exp(-2.0), 1e-9, label="bound at n=100, eps=0.1")
    almost(f(400, 0.05), 2 * math.exp(-2.0), 1e-9, label="bound at n=400, eps=0.05")
    almost(f(1, 0.1), 1.0, 1e-12, label="bound at n=1, eps=0.1 (2e^-0.02 = 1.96, capped at 1)")


@check("deviation_frequency is about 0.035 at n = 100, and the bound holds but is loose: 4 times the frequency or more")
def test_bound_loose(sub):
    bound = has(sub, "hoeffding_bound")
    freq = has(sub, "deviation_frequency")
    got = freq(100, 0.1, 20000, 0)
    almost(got, 0.0354, 0.01, label="simulated chance of missing 0.5 by more than 0.1")
    equal(freq(100, 0.1, 20000, 0), got, label="same seed, same answer")
    for n in (60, 100, 140):
        b = bound(n, 0.1)
        f = freq(n, 0.1, 40000, 1)
        if f > b:
            fail(f"at n={n} the simulated frequency {f:.4f} exceeds the bound {b:.4f}")
        if b < 4 * f:
            fail(f"at n={n} expected a loose bound (>= 4 x frequency), got bound {b:.4f} and frequency {f:.4f}")


@check("union_bound_sample_size grows with log k: 738, 1199, 2120 for k = 1, 10, 1000 at eps = delta = 0.05")
def test_sample_size(sub):
    # Plausible wrong implementation: log(2 / delta) / (2 eps^2), which drops k.
    # It returns 738 for every k, so the k = 1000 line fails.
    f = has(sub, "union_bound_sample_size")
    equal(int(f(1, 0.05, 0.05)), 738, label="n for k=1")
    equal(int(f(10, 0.05, 0.05)), 1199, label="n for k=10")
    equal(int(f(1000, 0.05, 0.05)), 2120, label="n for k=1000")
    ratio = f(10**6, 0.05, 0.05) / f(1, 0.05, 0.05)
    if not 4.5 < ratio < 5.0:
        fail(f"a million models should cost under 5 times the data of one model, got ratio {ratio:.2f}")


@check("union_bound_eps inverts union_bound_sample_size")
def test_eps_roundtrip(sub):
    size = has(sub, "union_bound_sample_size")
    eps_of = has(sub, "union_bound_eps")
    almost(eps_of(1000, 2120, 0.05), 0.04998, 1e-4, label="gap for k=1000, n=2120")
    for k, eps in ((5, 0.1), (200, 0.03), (10**4, 0.07)):
        n = size(k, eps, 0.1)
        got = eps_of(k, n, 0.1)
        if got > eps + 1e-12 or got < eps * 0.98:
            fail(f"k={k}: n={n} should give a gap just under {eps}, got {got:.5f}")


@check("shatters and num_dichotomies: thresholds shatter one point and not two")
def test_shatters_thresholds(sub):
    classes = has(sub, "threshold_class")(np.arange(0.5, 4.0, 1.0))
    shatters = has(sub, "shatters")
    count = has(sub, "num_dichotomies")
    one = np.array([[2.0]])
    two = np.array([[2.0], [3.0]])
    equal(count(two, classes), 3, label="label patterns thresholds make on two points")
    if not shatters(one, classes):
        fail("thresholds should shatter one point: both labels 0 and 1 are reachable")
    if shatters(two, classes):
        fail("thresholds make 3 patterns on two points, so they must not shatter them")


@check("largest_shattered finds VC dimension 1 for thresholds, 2 for intervals, 3 for half-planes")
def test_vc(sub):
    largest = has(sub, "largest_shattered")
    line = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    grid = np.arange(0.5, 6.0, 1.0)
    equal(largest(line, has(sub, "threshold_class")(grid)), 1, label="VC of thresholds")
    equal(largest(line, has(sub, "interval_class")(grid)), 2, label="VC of intervals")
    plane = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
    hp = has(sub, "halfplane_class")(48, np.linspace(-3.0, 3.0, 41))
    equal(largest(plane, hp), 3, label="VC of half-planes on five points in the plane")


@check("perceptron makes 3 mistakes on the chapter's four points and separates them")
def test_perceptron_hand(sub):
    # Plausible wrong implementation: update only when y * (w . x) < 0 (strict).
    # From w = 0 the product is 0, so it never updates, returns w = 0 and
    # reports 0 mistakes. A second wrong version updates on every row.
    f = has(sub, "perceptron")
    X = np.array([[1.0, 1.0], [1.0, -2.0], [-1.0, -2.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, -1.0, 1.0])
    w, mistakes = f(X, y)
    equal(int(mistakes), 3, label="mistakes on the hand example")
    close(np.asarray(w), np.array([3.0, 1.0]), atol=1e-12, label="final weights")


@check("perceptron mistakes stay at or under (R / margin)^2 on 10 separable problems")
def test_perceptron_bound(sub):
    perceptron = has(sub, "perceptron")
    bound = has(sub, "perceptron_mistake_bound")
    almost(
        bound(np.array([[1.0, 1.0], [1.0, -2.0], [-1.0, -2.0], [1.0, 2.0]]),
              np.array([1.0, 1.0, -1.0, 1.0]), np.array([1.0, 0.0])),
        5.0, 1e-9, label="bound on the hand example (R^2 = 5, margin = 1)")
    rng = np.random.default_rng(0)
    w_star = np.array([1.0, -0.5, 0.25])
    for trial in range(10):
        X = rng.normal(size=(400, 3))
        X = X[np.abs(X @ w_star) / np.linalg.norm(w_star) > 0.1 + 0.05 * trial][:120]
        y = np.sign(X @ w_star)
        w, mistakes = perceptron(X, y)
        b = bound(X, y, w_star)
        if np.any(y * (X @ w) <= 0):
            fail(f"trial {trial}: the final weights still misclassify a point")
        if mistakes > b:
            fail(f"trial {trial}: {mistakes} mistakes exceeds the bound {b:.1f}")


@check("min_norm_fit interpolates when p > n and returns the shortest such weights")
def test_min_norm(sub):
    # Plausible wrong implementation: solve (F^T F) w = F^T y. When p > n,
    # F^T F is singular, so the solve raises or returns garbage.
    fit = has(sub, "min_norm_fit")
    rng = np.random.default_rng(0)
    F = rng.normal(size=(8, 20))
    y = rng.normal(size=8)
    w = np.asarray(fit(F, y))
    close(F @ w, y, atol=1e-8, label="F w should equal y when p > n")
    null = np.linalg.svd(F)[2][-1]
    if np.linalg.norm(w) > np.linalg.norm(w + 0.5 * null) - 1e-9:
        fail("adding a null-space direction shortened the weights, so w was not the minimum-norm solution")
    A = rng.normal(size=(30, 4))
    b = rng.normal(size=30)
    close(np.asarray(fit(A, b)), np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-8, label="p < n: ordinary least squares")


@check("min_norm_test_error spikes at p = n = 30, then falls below the p = 15 level by p = 300")
def test_double_descent(sub):
    f = has(sub, "min_norm_test_error")
    low = f(30, 15, trials=40, seed=0)
    peak = f(30, 30, trials=40, seed=0)
    high = f(30, 300, trials=40, seed=0)
    if peak < 8 * low:
        fail(f"expected the error at p = 30 ({peak:.2f}) to be at least 8 times the error at p = 15 ({low:.2f})")
    if high > low:
        fail(f"expected the error at p = 300 ({high:.2f}) to fall below the error at p = 15 ({low:.2f})")
    sweep = {p: f(30, p, trials=40, seed=0) for p in (20, 25, 30, 35, 45)}
    top = max(sweep, key=sweep.get)
    if top != 30:
        fail(f"expected the peak of the sweep {sweep} at p = 30, found it at p = {top}")
