"""Checks for the bootstrap and bagging exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


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


@check("bootstrap_indices draws every row with equal chance")
def test_bootstrap_uniform(sub):
    got = has(sub, "bootstrap_indices")
    rng = np.random.default_rng(1)
    counts = np.zeros(4)
    for _ in range(2000):
        counts += np.bincount(got(4, rng), minlength=4)
    close(counts / counts.sum(), np.full(4, 0.25), atol=0.02, label="share of draws per row, n=4")


@check("the in-bag fraction of a bootstrap draw averages 1 - 1/e")
def test_in_bag_fraction(sub):
    got = has(sub, "bootstrap_indices")
    rng = np.random.default_rng(2)
    n = 2000
    fractions = [len(np.unique(got(n, rng))) / n for _ in range(60)]
    almost(float(np.mean(fractions)), 1.0 - np.exp(-1.0), tol=0.01, label="mean in-bag fraction at n=2000")
    small = [len(np.unique(got(4, rng))) / 4 for _ in range(4000)]
    almost(float(np.mean(small)), 1.0 - 0.75 ** 4, tol=0.015, label="mean in-bag fraction at n=4")


@check("fit_bagged_stumps keeps one stump and one full-size draw per estimator")
def test_fit_structure(sub):
    fit = has(sub, "fit_bagged_stumps")
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
    for stump in stumps:
        for key in ("feature", "threshold", "left_value", "right_value"):
            if key not in stump:
                fail(f"a stump is missing the key {key!r}")


@check("each stump is the stump fit to the rows in its own draw")
def test_fit_matches_draw(sub):
    fit = has(sub, "fit_bagged_stumps")
    fit_stump = has(sub, "_fit_stump")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    y = 2.0 * X[:, 1] + 0.2 * rng.normal(size=60)
    stumps, in_bag = fit(X, y, n_estimators=6, rng=rng)
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
