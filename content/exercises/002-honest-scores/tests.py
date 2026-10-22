"""Checks for the honest-scores exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape


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


@check("train_test_split repeats for one seed and shuffles for different seeds")
def test_split_seed(sub):
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


@check("train_test_split rounds the test count down, and 0.29 of 100 gives 29")
def test_split_count(sub):
    split = has(sub, "train_test_split")
    x = np.arange(100, dtype=float)
    _, _, x_test, _ = split(x, x, 0.29, 0)
    equal(len(x_test), 29, label="test count for 100 flats at 0.29")
    _, _, x_test, _ = split(np.arange(21, dtype=float), np.arange(21, dtype=float), 0.25, 0)
    equal(len(x_test), 5, label="test count for 21 flats at 0.25 (5.25 rounds down)")


@check("memoriser_predict copies the rent of the nearest training size")
def test_memoriser(sub):
    predict = has(sub, "memoriser_predict")
    x_train = np.array([3.0, 0.0, 1.0])  # deliberately unsorted
    y_train = np.array([7.0, 5.0, 6.0])
    got = predict(x_train, y_train, np.array([0.4, 0.6, 1.9, 2.1, 10.0, -4.0]))
    shape(got, (6,), label="memoriser_predict output")
    close(got, np.array([5.0, 6.0, 6.0, 7.0, 7.0, 5.0]), atol=0.0, label="nearest-size rents")


@check("memoriser_predict reproduces its own training rents exactly")
def test_memoriser_training(sub):
    predict = has(sub, "memoriser_predict")
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
