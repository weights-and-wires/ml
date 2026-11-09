"""Checks for the averaging and bagging exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape

def formula(rho, sigma2, B):
    return rho * sigma2 + (1 - rho) * sigma2 / B

X_HAND = np.array([[1.0], [2.0], [3.0], [4.0]])

Y_HAND = np.array([0.0, 10.0, 10.0, 0.0])

STUMPS_HAND = [
    {"feature": 0, "threshold": 1.5, "left_value": 0.0, "right_value": 10.0},
    {"feature": 0, "threshold": 3.5, "left_value": 10.0, "right_value": 0.0},
]

DRAWS_HAND = [np.array([0, 0, 1, 1]), np.array([2, 2, 3, 3])]

def constant_stump(value):
    return {"feature": None, "threshold": None, "left_value": value, "right_value": value}

STUMPS_CONST = [constant_stump(3.0), constant_stump(6.0), constant_stump(9.0)]

DRAWS_CONST = [np.array([0, 0, 1, 1]), np.array([1, 2, 2, 2]), np.array([0, 1, 2, 3])]


def _part_bagging_variance_extremes(sub):
    got = has(sub, "bagging_variance")
    almost(got(0.0, 2.0, 10), 0.2, tol=1e-10, label="rho=0, sigma2=2, B=10")
    almost(got(1.0, 2.0, 10), 2.0, tol=1e-10, label="rho=1, sigma2=2, B=10")
    almost(got(0.4, 3.0, 1), 3.0, tol=1e-10, label="B=1 returns one model's variance")
    almost(got(0.5, 4.0, 2), 3.0, tol=1e-10, label="rho=0.5, sigma2=4, B=2")


def _part_bagging_variance_floor(sub):
    got = has(sub, "bagging_variance")
    almost(got(0.3, 1.0, 50), 0.314, tol=1e-10, label="rho=0.3, sigma2=1, B=50")
    almost(got(0.3, 1.0, 5000), 0.30014, tol=1e-10, label="rho=0.3, sigma2=1, B=5000")
    almost(got(0.8, 1.0, 10**7), 0.8, tol=1e-6, label="B huge approaches rho * sigma2")


def _part_estimators_shape(sub):
    gen = has(sub, "correlated_estimators")
    a = gen(0.4, 2.0, 7, 30, np.random.default_rng(0))
    b = gen(0.4, 2.0, 7, 30, np.random.default_rng(0))
    shape(a, (30, 7), label="correlated_estimators output")
    if not np.allclose(a, b):
        fail("two calls with the same seed gave different arrays, so the draws are not coming from rng")
    if np.allclose(a[:, 0], a[:, 1]):
        fail("columns 0 and 1 are identical, but rho=0.4 should leave each column its own private part")


def _part_estimators_variance(sub):
    gen = has(sub, "correlated_estimators")
    rng = np.random.default_rng(1)
    sigma2 = 2.5
    for rho in (0.0, 0.3, 0.8):
        draws = gen(rho, sigma2, 4, 40000, rng)
        for b in range(4):
            almost(float(draws[:, b].var()), sigma2, tol=0.06 * sigma2, label=f"variance of column {b} at rho={rho}")
        almost(float(draws.mean()), 0.0, tol=0.03, label=f"mean at rho={rho}")


def _part_estimators_correlation(sub):
    gen = has(sub, "correlated_estimators")
    rng = np.random.default_rng(2)
    for rho in (0.0, 0.3, 0.8, 1.0):
        draws = gen(rho, 1.7, 4, 40000, rng)
        corr = np.corrcoef(draws, rowvar=False)
        off = corr[~np.eye(4, dtype=bool)]
        for value in off:
            almost(float(value), rho, tol=0.03, label=f"pairwise correlation at rho={rho}")


def _part_trees_needed(sub):
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


def _part_trees_needed_edges(sub):
    need = has(sub, "trees_needed")
    if need(0.5, 1.0, 100.0) != 1:
        fail(f"a huge tolerance needs only 1 model, got {need(0.5, 1.0, 100.0)}")
    try:
        need(0.0, 1.0, 0.05)
    except ValueError:
        return
    fail("trees_needed(0.0, ...) should raise ValueError, because the floor is 0")


def _part_bootstrap_shape(sub):
    got = has(sub, "bootstrap_indices")
    idx = got(50, np.random.default_rng(0))
    shape(idx, (50,), label="bootstrap_indices output")
    if not np.issubdtype(np.asarray(idx).dtype, np.integer):
        fail(f"indices must be integers, got dtype {np.asarray(idx).dtype}")
    if idx.min() < 0 or idx.max() > 49:
        fail(f"indices must lie in [0, 49], got range [{idx.min()}, {idx.max()}]")
    if len(np.unique(idx)) == 50:
        fail("50 draws from 50 rows were all distinct. A bootstrap draw has repeats, and some rows are missing")


def _part_bootstrap_uniform_and_fraction(sub):
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


def _part_bagged_predict_hand(sub):
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


def _part_bagging_lowers_variance(sub):
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


def _part_oob_error_hand(sub):
    got = has(sub, "out_of_bag_error")(X_HAND, Y_HAND, STUMPS_HAND, DRAWS_HAND)
    # Averaging both stumps on every row would give predictions [5, 10, 10, 5] and an error of 12.5.
    almost(got, 50.0, tol=1e-9, label="out-of-bag mean squared error")


def _part_oob_error_uncovered(sub):
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


def _part_oob_tracks_test(sub):
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


@check("bagging_variance matches its extremes and the worked examples, and keeps the shared part whole as B grows")
def test_bagging_variance_extremes_and_bagging_variance_floor(sub):
    _part_bagging_variance_extremes(sub)
    _part_bagging_variance_floor(sub)


@check("correlated_estimators returns an (n_draws, B) array, repeats under one seed, and gives every column variance sigma2 and every pair correlation rho")
def test_estimators_shape_and_estimators_variance_and_estimators_correlation(sub):
    _part_estimators_shape(sub)
    _part_estimators_variance(sub)
    _part_estimators_correlation(sub)


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


@check("trees_needed returns the smallest B within the tolerance of the floor, returns 1 for a huge tolerance, and rejects rho = 0")
def test_trees_needed_and_trees_needed_edges(sub):
    _part_trees_needed(sub)
    _part_trees_needed_edges(sub)


@check("bootstrap_indices returns n integers in range with repeats and gaps, draws every row with equal chance, and averages an in-bag fraction of 1 - 1/e")
def test_bootstrap_shape_and_bootstrap_uniform_and_fraction(sub):
    _part_bootstrap_shape(sub)
    _part_bootstrap_uniform_and_fraction(sub)


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


@check("bagged_predict averages the stumps' predictions, and bagging lowers the variance of a stump's prediction")
def test_bagged_predict_hand_and_bagging_lowers_variance(sub):
    _part_bagged_predict_hand(sub)
    _part_bagging_lowers_variance(sub)


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


@check("out_of_bag_error is 50 on the hand example, skips rows no stump missed, rejects the case with none, and tracks the error on a fresh test set")
def test_oob_error_hand_and_oob_error_uncovered_and_oob_tracks_test(sub):
    _part_oob_error_hand(sub)
    _part_oob_error_uncovered(sub)
    _part_oob_tracks_test(sub)
