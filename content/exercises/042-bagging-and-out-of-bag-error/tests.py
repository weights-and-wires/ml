"""Checks for the bagging and out-of-bag error exercise."""

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


@check("bootstrap_indices returns n integers in range, with repeats and gaps")
def test_bootstrap_shape(sub):
    got = has(sub, "bootstrap_indices")
    idx = got(50, np.random.default_rng(0))
    shape(idx, (50,), label="bootstrap_indices output")
    if not np.issubdtype(np.asarray(idx).dtype, np.integer):
        fail(f"indices must be integers, got dtype {np.asarray(idx).dtype}")
    if idx.min() < 0 or idx.max() > 49:
        fail(f"indices must lie in [0, 49], got range [{idx.min()}, {idx.max()}]")
    if len(np.unique(idx)) == 50:
        fail("50 draws from 50 rows were all distinct. A bootstrap draw has repeats, and some rows are missing")


@check("bootstrap_indices draws every row with equal chance, and the in-bag fraction averages 1 - 1/e")
def test_bootstrap_uniform_and_fraction(sub):
    got = has(sub, "bootstrap_indices")
    rng = np.random.default_rng(1)
    counts = np.zeros(4)
    for _ in range(2000):
        counts += np.bincount(got(4, rng), minlength=4)
    close(counts / counts.sum(), np.full(4, 0.25), atol=0.02, label="share of draws per row, n=4")
    rng = np.random.default_rng(2)
    n = 2000
    fractions = [len(np.unique(got(n, rng))) / n for _ in range(60)]
    almost(float(np.mean(fractions)), 1.0 - np.exp(-1.0), tol=0.01, label="mean in-bag fraction at n=2000")
    small = [len(np.unique(got(4, rng))) / 4 for _ in range(4000)]
    almost(float(np.mean(small)), 1.0 - 0.75 ** 4, tol=0.015, label="mean in-bag fraction at n=4")


@check("fit_bagged_stumps keeps one full-size draw per stump, and each stump is fit to its own draw")
def test_fit_bagged(sub):
    fit = has(sub, "fit_bagged_stumps")
    fit_stump = has(sub, "_fit_stump")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=100)
    stumps, in_bag = fit(X, y, n_estimators=15, rng=rng)
    if len(stumps) != 15 or len(in_bag) != 15:
        fail(f"expected 15 stumps and 15 index arrays, got {len(stumps)} and {len(in_bag)}")
    for idx in in_bag:
        shape(np.asarray(idx), (100,), label="one bootstrap index array")
    if all(np.array_equal(in_bag[0], idx) for idx in in_bag[1:]):
        fail("all 15 index arrays are identical, so every stump saw the same draw")
    for b, (stump, idx) in enumerate(zip(stumps, in_bag)):
        want = fit_stump(X[idx], y[idx])
        if stump["feature"] != want["feature"]:
            fail(f"stump {b} splits on feature {stump['feature']}, but a stump fit to its draw splits on {want['feature']}")
        close(
            [stump["threshold"], stump["left_value"], stump["right_value"]],
            [want["threshold"], want["left_value"], want["right_value"]],
            atol=1e-9,
            label=f"stump {b} threshold and leaf values",
        )


@check("bagged_predict averages the stumps' predictions")
def test_bagged_predict_hand(sub):
    predict = has(sub, "bagged_predict")
    stumps = [
        {"feature": 0, "threshold": 1.5, "left_value": 0.0, "right_value": 10.0},
        {"feature": 0, "threshold": 3.5, "left_value": 10.0, "right_value": 0.0},
    ]
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = predict(stumps, X)
    shape(out, (4,), label="bagged_predict output")
    close(out, np.array([5.0, 10.0, 10.0, 5.0]), atol=1e-12, label="mean of the two stumps")
    close(predict(stumps[:1], X), np.array([0.0, 10.0, 10.0, 10.0]), atol=1e-12, label="one stump alone")


@check("bagging lowers the variance of a stump's prediction across training sets")
def test_bagging_lowers_variance(sub):
    fit = has(sub, "fit_bagged_stumps")
    predict = has(sub, "bagged_predict")
    fit_stump = has(sub, "_fit_stump")
    predict_stump = has(sub, "_predict_stump")
    rng = np.random.default_rng(11)
    grid = np.linspace(0.05, 0.95, 19)[:, None]
    single, bagged = [], []
    for _ in range(40):
        X = rng.uniform(0, 1, size=(30, 1))
        y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.normal(size=30)
        single.append(predict_stump(fit_stump(X, y), grid))
        stumps, _ = fit(X, y, n_estimators=15, rng=rng)
        bagged.append(predict(stumps, grid))
    var_single = float(np.var(single, axis=0).mean())
    var_bagged = float(np.var(bagged, axis=0).mean())
    if var_bagged > 0.85 * var_single:
        fail(
            f"bagged variance {var_bagged:.4f} should be clearly below the single-stump variance "
            f"{var_single:.4f}, because each stump puts its split in a different place"
        )



@check("oob_mask marks the rows a draw never picked, and nothing else")
def test_oob_mask(sub):
    got = has(sub, "oob_mask")
    mask = got(np.array([0, 0, 2, 3]), 4)
    shape(mask, (4,), label="oob_mask output")
    if np.asarray(mask).dtype != bool:
        fail(f"oob_mask must return a boolean array, got dtype {np.asarray(mask).dtype}")
    equal(mask, np.array([False, True, False, False]), label="mask for the draw [0, 0, 2, 3]")
    rng = np.random.default_rng(0)
    n = 2000
    idx = rng.integers(0, n, size=n)
    equal(got(idx, n), ~np.isin(np.arange(n), idx), label="mask for a random draw")
    almost(float(got(idx, n).mean()), 0.368, tol=0.04, label="fraction out of the bag")
    if got(np.arange(7), 7).any():
        fail("when the draw contains every row, no row is out of the bag")


@check("oob_predictions averages only the stumps that missed each row")
def test_oob_predictions(sub):
    got = has(sub, "oob_predictions")
    pred, count = got(X_HAND, STUMPS_CONST, DRAWS_CONST)
    shape(pred, (4,), label="pred")
    shape(count, (4,), label="count")
    close(count, np.array([1.0, 0.0, 1.0, 2.0]), atol=1e-12, label="number of stumps that missed each row")
    close(pred[[0, 2, 3]], np.array([6.0, 3.0, 4.5]), atol=1e-12, label="mean over missing stumps, rows 0, 2, 3")
    if not np.isnan(pred[1]):
        fail(f"row 1 is in every bag, so its prediction should be nan, got {pred[1]}")
    pred, count = got(X_HAND, STUMPS_HAND, DRAWS_HAND)
    close(pred, np.full(4, 10.0), atol=1e-12, label="held-out predictions on the hand example")
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
