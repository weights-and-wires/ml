"""Checks for the scoring-a-guess exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

GUESSES = np.array([220.0, 330.0, 440.0, 550.0])  # the rule "eleven times the size"
TARGETS = np.array([200.0, 300.0, 400.0, 650.0])
SIZES = np.array([20.0, 30.0, 40.0, 50.0])
RENTS = np.array([200.0, 300.0, 400.0, 650.0])


@check("squared_loss returns one score per example")
def test_squared_loss(sub):
    squared_loss = has(sub, "squared_loss")
    out = squared_loss(GUESSES, TARGETS)
    shape(out, (4,), label="squared_loss output")
    close(out, np.array([400.0, 900.0, 1600.0, 10000.0]), atol=1e-9, label="squared misses of the four flats")
    # A miss above the target and a miss below it must cost the same.
    close(squared_loss(np.array([3.0, 1.0]), np.array([2.0, 2.0])), np.array([1.0, 1.0]), atol=1e-12,
          label="+1 and -1 misses")


@check("empirical_risk averages the squared loss: 3225 on the four-flat example")
def test_empirical_risk_squared(sub):
    risk = has(sub, "empirical_risk")
    squared = lambda p, t: (p - t) ** 2
    almost(risk(GUESSES, TARGETS, squared), 3225.0, tol=1e-9, label="mean squared loss")
    # Doubling the number of flats must not change the average.
    almost(risk(np.tile(GUESSES, 2), np.tile(TARGETS, 2), squared), 3225.0, tol=1e-9,
           label="mean over eight flats (the same four, twice)")


@check("empirical_risk returns a Python float and works with any per-example loss")
def test_empirical_risk_other_losses(sub):
    risk = has(sub, "empirical_risk")
    zero_one = lambda p, t: (p != t).astype(float)
    absolute = lambda p, t: np.abs(p - t)
    got = risk(np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]), zero_one)
    if type(got) is not float:
        fail(f"empirical_risk must return a Python float, got {type(got).__name__}")
    almost(got, 2.0 / 3.0, tol=1e-12, label="mean zero-one loss")
    almost(risk(GUESSES, TARGETS, absolute), 47.5, tol=1e-9, label="mean absolute loss")


@check("best_slope returns 11 under squared loss on the four flats")
def test_best_slope_squared(sub):
    best_slope = has(sub, "best_slope")
    squared = lambda p, t: (p - t) ** 2
    got = best_slope(SIZES, RENTS, np.arange(6, 15, dtype=float), squared)
    if got != 11.0:
        fail(f"expected slope 11 (mean squared loss 3225, against 5625 at slope 10), got {got}")


@check("best_slope returns 10 under absolute loss, because outliers count less")
def test_best_slope_absolute(sub):
    best_slope = has(sub, "best_slope")
    absolute = lambda p, t: np.abs(p - t)
    got = best_slope(SIZES, RENTS, np.arange(6, 15, dtype=float), absolute)
    if got != 10.0:
        fail(f"expected slope 10 (mean absolute loss 37.5, against 47.5 at slope 11), got {got}")


@check("best_slope agrees with a brute-force scan on seeded random flats")
def test_best_slope_random(sub):
    best_slope = has(sub, "best_slope")
    rng = np.random.default_rng(0)
    sizes = rng.uniform(15.0, 80.0, size=25)
    rents = 9.0 * sizes + rng.normal(scale=30.0, size=25)
    slopes = np.linspace(5.0, 14.0, 37)
    squared = lambda p, t: (p - t) ** 2
    risks = np.array([np.mean(squared(s * sizes, rents)) for s in slopes])
    want = float(slopes[np.argmin(risks)])
    got = best_slope(sizes, rents, slopes, squared)
    almost(got, want, tol=1e-12, label="best slope")
    if type(got) is not float:
        fail(f"best_slope must return a Python float, got {type(got).__name__}")
