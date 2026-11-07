"""Checks for the loss, risk, and honest-scores exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape

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


@check("empirical_risk averages any loss and returns a Python float: 3225 on the four flats, 2/3 on the spam example")
def test_empirical_risk(sub):
    risk = has(sub, "empirical_risk")
    squared = lambda p, t: (p - t) ** 2
    absolute = lambda p, t: np.abs(p - t)
    zero_one = lambda p, t: (p != t).astype(float)
    almost(risk(GUESSES, TARGETS, squared), 3225.0, tol=1e-9, label="mean squared loss")
    # Doubling the number of flats must not change the average.
    almost(risk(np.tile(GUESSES, 2), np.tile(TARGETS, 2), squared), 3225.0, tol=1e-9,
           label="mean over eight flats (the same four, twice)")
    almost(risk(GUESSES, TARGETS, absolute), 47.5, tol=1e-9, label="mean absolute loss")
    got = risk(np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]), zero_one)
    if type(got) is not float:
        fail(f"empirical_risk must return a Python float, got {type(got).__name__}")
    almost(got, 2.0 / 3.0, tol=1e-12, label="mean zero-one loss")


@check("best_slope returns 11 under squared loss and 10 under absolute loss, and matches a brute-force scan")
def test_best_slope(sub):
    best_slope = has(sub, "best_slope")
    squared = lambda p, t: (p - t) ** 2
    absolute = lambda p, t: np.abs(p - t)
    grid = np.arange(6, 15, dtype=float)
    got = best_slope(SIZES, RENTS, grid, squared)
    if got != 11.0:
        fail(f"expected slope 11 under squared loss (mean 3225, against 5625 at slope 10), got {got}")
    got = best_slope(SIZES, RENTS, grid, absolute)
    if got != 10.0:
        fail(f"expected slope 10 under absolute loss (mean 37.5, against 47.5 at slope 11), got {got}")
    rng = np.random.default_rng(0)
    sizes = rng.uniform(15.0, 80.0, size=25)
    rents = 9.0 * sizes + rng.normal(scale=30.0, size=25)
    slopes = np.linspace(5.0, 14.0, 37)
    risks = np.array([np.mean(squared(s * sizes, rents)) for s in slopes])
    want = float(slopes[np.argmin(risks)])
    got = best_slope(sizes, rents, slopes, squared)
    almost(got, want, tol=1e-12, label="best slope on seeded random flats")
    if type(got) is not float:
        fail(f"best_slope must return a Python float, got {type(got).__name__}")


@check("train_test_split keeps every flat once, with each size paired to its rent")
def test_split_partition(sub):
    split = has(sub, "train_test_split")
    x = np.arange(20, dtype=float)
    y = 2.0 * x + 1.0
    x_train, y_train, x_test, y_test = split(x, y, 0.25, 0)
    equal((len(x_train), len(x_test)), (15, 5), label="(training count, test count) for 20 flats at 0.25")
    close(y_train, 2.0 * x_train + 1.0, atol=0.0, label="training rents follow training sizes")
    close(y_test, 2.0 * x_test + 1.0, atol=0.0, label="test rents follow test sizes")
    close(np.sort(np.concatenate([x_train, x_test])), x, atol=0.0, label="every flat appears exactly once")


@check("train_test_split repeats for one seed, shuffles for different seeds, and rounds the test count down")
def test_split_seed_and_count(sub):
    split = has(sub, "train_test_split")
    x = np.arange(40, dtype=float)
    y = x * 3.0
    a = split(x, y, 0.25, 7)
    b = split(x, y, 0.25, 7)
    c = split(x, y, 0.25, 8)
    close(a[2], b[2], atol=0.0, label="same seed gives the same test flats")
    if np.array_equal(a[2], c[2]):
        fail("different seeds gave identical test sets; the shuffle must depend on the seed")
    if np.array_equal(a[2], x[-10:]) or np.array_equal(a[2], x[:10]):
        fail("the test set is the last or first rows; shuffle the positions before you cut")
    x = np.arange(100, dtype=float)
    _, _, x_test, _ = split(x, x, 0.29, 0)
    equal(len(x_test), 29, label="test count for 100 flats at 0.29")
    _, _, x_test, _ = split(np.arange(21, dtype=float), np.arange(21, dtype=float), 0.25, 0)
    equal(len(x_test), 5, label="test count for 21 flats at 0.25 (5.25 rounds down)")


@check("memoriser_predict copies the rent of the nearest training size, and reproduces its own training rents")
def test_memoriser(sub):
    predict = has(sub, "memoriser_predict")
    x_train = np.array([3.0, 0.0, 1.0])  # deliberately unsorted
    y_train = np.array([7.0, 5.0, 6.0])
    got = predict(x_train, y_train, np.array([0.4, 0.6, 1.9, 2.1, 10.0, -4.0]))
    shape(got, (6,), label="memoriser_predict output")
    close(got, np.array([5.0, 6.0, 6.0, 7.0, 7.0, 5.0]), atol=0.0, label="nearest-size rents")
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=30)
    y = rng.normal(size=30)
    close(predict(x, y, x), y, atol=0.0, label="guesses on the training flats")


@check("generalisation_gap matches an independent computation")
def test_gap(sub):
    gap_fn = has(sub, "generalisation_gap")
    split = has(sub, "train_test_split")
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=20)
    y = np.sin(2.0 * np.pi * x) + 0.3 * rng.normal(size=20)
    got = gap_fn(x, y, 0.25, 0)
    if len(got) != 3:
        fail(f"expected (train_risk, test_risk, gap), got {len(got)} values")
    train_risk, test_risk, gap = got
    x_tr, y_tr, x_te, y_te = split(x, y, 0.25, 0)
    nearest = np.argmin(np.abs(x_te[:, None] - x_tr[None, :]), axis=1)
    want = float(np.mean((y_tr[nearest] - y_te) ** 2))
    almost(train_risk, 0.0, tol=1e-12, label="training risk of the memoriser")
    almost(test_risk, want, tol=1e-12, label="test risk")
    almost(gap, want, tol=1e-12, label="gap = test risk - training risk")
    for value in got:
        if type(value) is not float:
            fail(f"every returned value must be a Python float, got {type(value).__name__}")
