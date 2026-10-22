"""Checks for the out-of-bag error exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape

# The hand example: x = [1, 2, 3, 4], y = [0, 10, 10, 0], two stumps.
X_HAND = np.array([[1.0], [2.0], [3.0], [4.0]])
Y_HAND = np.array([0.0, 10.0, 10.0, 0.0])
STUMPS_HAND = [
    {"feature": 0, "threshold": 1.5, "left_value": 0.0, "right_value": 10.0},
    {"feature": 0, "threshold": 3.5, "left_value": 10.0, "right_value": 0.0},
]
DRAWS_HAND = [np.array([0, 0, 1, 1]), np.array([2, 2, 3, 3])]


def constant_stump(value):
    return {"feature": None, "threshold": None, "left_value": value, "right_value": value}


# Three constant stumps, each missing a different set of rows.
STUMPS_CONST = [constant_stump(3.0), constant_stump(6.0), constant_stump(9.0)]
DRAWS_CONST = [np.array([0, 0, 1, 1]), np.array([1, 2, 2, 2]), np.array([0, 1, 2, 3])]


@check("oob_mask marks the rows a draw never picked")
def test_oob_mask_hand(sub):
    mask = has(sub, "oob_mask")(np.array([0, 0, 2, 3]), 4)
    shape(mask, (4,), label="oob_mask output")
    if np.asarray(mask).dtype != bool:
        fail(f"oob_mask must return a boolean array, got dtype {np.asarray(mask).dtype}")
    equal(mask, np.array([False, True, False, False]), label="mask for the draw [0, 0, 2, 3]")


@check("oob_mask is the exact complement of the rows in the draw")
def test_oob_mask_random(sub):
    got = has(sub, "oob_mask")
    rng = np.random.default_rng(0)
    n = 2000
    idx = rng.integers(0, n, size=n)
    equal(got(idx, n), ~np.isin(np.arange(n), idx), label="mask for a random draw")
    almost(float(got(idx, n).mean()), 0.368, tol=0.04, label="fraction out of the bag")
    if got(np.arange(7), 7).any():
        fail("when the draw contains every row, no row is out of the bag")


@check("oob_predictions averages only the stumps that missed each row")
def test_oob_predictions_average(sub):
    pred, count = has(sub, "oob_predictions")(X_HAND, STUMPS_CONST, DRAWS_CONST)
    shape(pred, (4,), label="pred")
    shape(count, (4,), label="count")
    close(count, np.array([1.0, 0.0, 1.0, 2.0]), atol=1e-12, label="number of stumps that missed each row")
    close(pred[[0, 2, 3]], np.array([6.0, 3.0, 4.5]), atol=1e-12, label="mean over missing stumps, rows 0, 2, 3")
    if not np.isnan(pred[1]):
        fail(f"row 1 is in every bag, so its prediction should be nan, got {pred[1]}")


@check("oob_predictions on the hand example gives 10 for every row")
def test_oob_predictions_hand(sub):
    pred, count = has(sub, "oob_predictions")(X_HAND, STUMPS_HAND, DRAWS_HAND)
    close(pred, np.full(4, 10.0), atol=1e-12, label="held-out predictions")
    close(count, np.ones(4), atol=1e-12, label="one stump missed each row")


@check("out_of_bag_error on the hand example is 50, not the training error")
def test_oob_error_hand(sub):
    got = has(sub, "out_of_bag_error")(X_HAND, Y_HAND, STUMPS_HAND, DRAWS_HAND)
    # Averaging both stumps on every row would give predictions [5, 10, 10, 5] and an error of 12.5.
    almost(got, 50.0, tol=1e-9, label="out-of-bag mean squared error")


@check("out_of_bag_error skips rows that no stump missed, and rejects the case with none")
def test_oob_error_uncovered(sub):
    got = has(sub, "out_of_bag_error")
    y = np.zeros(4)
    value = got(X_HAND, y, STUMPS_CONST, DRAWS_CONST)
    if not np.isfinite(value):
        fail(f"row 1 has no out-of-bag stump and must be skipped, but the error is {value}")
    almost(value, (36.0 + 9.0 + 20.25) / 3.0, tol=1e-9, label="mean over the three scored rows")
    full = [np.arange(4), np.arange(4)]
    try:
        got(X_HAND, y, STUMPS_CONST[:2], full)
    except ValueError:
        return
    fail("every row is in every bag, so out_of_bag_error should raise ValueError")


@check("out-of-bag error tracks the error on a fresh test set")
def test_oob_tracks_test(sub):
    fit = has(sub, "fit_bagged_stumps")
    oob_fn = has(sub, "out_of_bag_error")
    predict_stump = has(sub, "_predict_stump")
    rng = np.random.default_rng(4)
    n = 500
    true_f = lambda X: np.maximum(X[:, 0], 0.0) - 0.5 * X[:, 1]
    X = rng.normal(size=(n, 3))
    y = true_f(X) + 0.3 * rng.normal(size=n)
    X_test = rng.normal(size=(n, 3))
    y_test = true_f(X_test) + 0.3 * rng.normal(size=n)

    stumps, in_bag = fit(X, y, n_estimators=80, rng=rng)
    oob = oob_fn(X, y, stumps, in_bag)
    ensemble = np.mean([predict_stump(s, X_test) for s in stumps], axis=0)
    test_mse = float(np.mean((ensemble - y_test) ** 2))
    if abs(oob - test_mse) > 0.1 * test_mse:
        fail(f"out-of-bag error {oob:.4f} should be within 10% of the test error {test_mse:.4f}")
