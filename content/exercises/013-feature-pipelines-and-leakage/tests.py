"""Checks for the feature pipelines and data leakage exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def noise_problem(seed, n=50, d=500):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.integers(0, 2, size=n)


SIX_CATS = np.array(list("AAABCC"))
SIX_Y = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])


@check("Standardiser stores the training mean and std, and [2, 4, 6, 8] maps 10 to 2.2361")
def test_standardiser_hand(sub):
    cls = has(sub, "Standardiser")
    train = np.array([[2.0], [4.0], [6.0], [8.0]])
    s = cls().fit(train)
    close(s.mean_, np.array([5.0]), atol=1e-12, label="stored mean")
    close(s.std_, np.array([np.sqrt(5.0)]), atol=1e-12, label="stored std (divide by n)")
    close(s.transform(np.array([[10.0]])), np.array([[5.0 / np.sqrt(5.0)]]), atol=1e-12,
          label="test value 10 under the training statistics")
    out = cls().fit_transform(train)
    close(out.mean(axis=0), np.zeros(1), atol=1e-12, label="column mean after fit_transform")
    close(out.std(axis=0), np.ones(1), atol=1e-12, label="column std after fit_transform")


@check("Standardiser.transform never reads test statistics: changing the test rows leaves the stored numbers alone")
def test_standardiser_no_test_stats(sub):
    cls = has(sub, "Standardiser")
    rng = np.random.default_rng(0)
    train = rng.normal(loc=3.0, scale=2.0, size=(40, 3))
    s = cls().fit(train)
    mean_before, std_before = s.mean_.copy(), s.std_.copy()
    test_a = rng.normal(size=(10, 3))
    test_b = test_a + 1000.0
    out_a, out_b = s.transform(test_a), s.transform(test_b)
    close(s.mean_, mean_before, atol=0, label="mean_ after transform")
    close(s.std_, std_before, atol=0, label="std_ after transform")
    close(out_b - out_a, np.full((10, 3), 1000.0) / std_before, rtol=1e-9,
          label="a shift of 1000 in the test rows must shift the output by 1000 / std_ and no more")
    const = np.column_stack([np.ones(5), np.arange(5.0)])
    out = cls().fit(const).transform(const)
    if not np.all(np.isfinite(out)):
        fail(f"a constant column should not divide by zero, got {out[:, 0]}")


@check("Imputer fills with the training mean or median and ignores NaN when fitting")
def test_imputer(sub):
    cls = has(sub, "Imputer")
    train = np.array([[1.0, 5.0], [2.0, 5.0], [np.nan, np.nan], [9.0, 5.0]])
    mean_imp = cls("mean").fit(train)
    median_imp = cls("median").fit(train)
    close(mean_imp.fill_, np.array([4.0, 5.0]), atol=1e-12, label="mean fill [1,2,nan,9] and [5,5,nan,5]")
    close(median_imp.fill_, np.array([2.0, 5.0]), atol=1e-12, label="median fill")
    test = np.array([[np.nan, 7.0], [3.0, np.nan]])
    snapshot = test.copy()
    got = mean_imp.transform(test)
    close(got, np.array([[4.0, 7.0], [3.0, 5.0]]), atol=1e-12, label="test rows after mean imputation")
    if not np.array_equal(test, snapshot, equal_nan=True):
        fail("transform changed its input; return a copy")
    big = np.array([[np.nan, 1e6]])
    close(mean_imp.transform(big)[0, 0], 4.0, atol=1e-12,
          label="filling uses the training mean, however extreme the test rows are")


@check("one_hot gives one column per fitted category plus an unknown bucket, and every row sums to 1")
def test_one_hot(sub):
    fn = has(sub, "one_hot")
    got = fn(["red", "green", "blue"], ["green", "purple", "red", "blue"])
    want = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    shape(got, (4, 4), label="one_hot output")
    close(got, want, atol=0, label="one_hot of green, purple, red, blue")
    close(got.sum(axis=1), np.ones(4), atol=0, label="row sums")
    close(fn(["a"], ["x", "y"]), np.array([[0.0, 1.0], [0.0, 1.0]]), atol=0,
          label="two unseen values share the one unknown bucket")


@check("TargetEncoder smooths toward the prior, and target_encode_oof never lets a row's own target into its encoding")
def test_target_encode_oof(sub):
    cls = has(sub, "TargetEncoder")
    enc = cls(smoothing=2.0).fit(SIX_CATS, SIX_Y)
    close(enc.prior_, 0.5, atol=1e-12, label="prior")
    close(enc.transform(np.array(["A", "B", "C", "Z"])), np.array([0.6, 1.0 / 3.0, 0.5, 0.5]),
          atol=1e-12, label="encodings of A, B, C and an unseen Z at smoothing 2")
    zero = cls(smoothing=0.0).fit(SIX_CATS, SIX_Y)
    close(zero.transform(np.array(["A", "B"])), np.array([2.0 / 3.0, 0.0]), atol=1e-12,
          label="smoothing 0 gives the plain category mean")
    fn = has(sub, "target_encode_oof")
    folds = [np.array([0, 3, 4]), np.array([1, 2, 5])]
    got = fn(SIX_CATS, SIX_Y, folds, 2.0)
    want = np.array([5 / 12, 7 / 9, 7 / 9, 1 / 3, 2 / 9, 7 / 9])
    close(got, want, atol=1e-12, label="out-of-fold encodings of the six rows at smoothing 2")
    rng = np.random.default_rng(0)
    cats = rng.integers(0, 6, size=60)
    y = rng.normal(size=60)
    folds = [np.arange(i, 60, 4) for i in range(4)]
    base = fn(cats, y, folds, 1.0)
    for i in (0, 7, 33):
        y2 = y.copy()
        y2[i] += 100.0
        moved = fn(cats, y2, folds, 1.0)
        if abs(moved[i] - base[i]) > 1e-12:
            fail(f"row {i}: changing its own target moved its encoding from {base[i]:.6g} to {moved[i]:.6g}")


@check("target_encode_oof of pure-noise labels is nearly uncorrelated with them, while an all-rows encoding is not")
def test_target_leak(sub):
    oof = has(sub, "target_encode_oof")
    enc = has(sub, "TargetEncoder")
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 150, size=300)          # about two rows per category
    y = rng.integers(0, 2, size=300).astype(float)  # pure noise
    folds = [np.arange(i, 300, 5) for i in range(5)]
    leaky = enc(1.0).fit(ids, y).transform(ids)
    leaky_corr = float(np.corrcoef(leaky, y)[0, 1])
    safe_corr = float(np.corrcoef(oof(ids, y, folds, 1.0), y)[0, 1])
    if leaky_corr < 0.4:
        fail(f"the all-rows encoding should correlate strongly with noise labels, got {leaky_corr:.3f}; check TargetEncoder")
    # Wrong implementation caught here: target_encode_oof that fits one TargetEncoder on all rows
    # and transforms every row with it. It reproduces leaky_corr (about 0.65) instead of near 0.
    if abs(safe_corr) > 0.2:
        fail(f"out-of-fold encoding of noise labels has correlation {safe_corr:.3f} with them, expected within 0.2 of 0")


@check("select_top_k returns the planted informative columns, sorted, and ignores the sign of the correlation")
def test_select_top_k(sub):
    fn = has(sub, "select_top_k")
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n).astype(float)
    X = rng.normal(size=(n, 20))
    X[:, 4] += 3.0 * y
    X[:, 11] -= 2.0 * y
    got = fn(X, y, 2)
    equal(np.asarray(got).tolist(), [4, 11], label="two planted columns, one positive and one negative")
    shape(np.asarray(fn(X, y, 5)), (5,), label="select_top_k(k=5)")


@check("cv_score refits inside each fold: the pipeline sees only training rows, and each row is scored once")
def test_cv_score(sub):
    fn = has(sub, "cv_score")
    n = 30
    X = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    y = (np.arange(n) % 2).astype(int)
    folds = [np.arange(i, n, 5) for i in range(5)]
    calls = []

    def pipeline_fit(Xt, yt):
        seen = set(Xt[:, 0].astype(int).tolist())
        calls.append(len(seen))

        def predict(Xs):
            ids = set(Xs[:, 0].astype(int).tolist())
            if seen & ids:
                fail(f"rows {sorted(seen & ids)} were in both the training and the held-out set")
            return np.zeros(len(Xs), dtype=int)

        return predict

    got = fn(pipeline_fit, X, y, folds)
    shape(got, (5,), label="cv_score output")
    equal(calls, [24] * 5, label="training-set size of each of the 5 refits")
    close(got, np.full(5, 0.5), atol=1e-12, label="accuracy of an all-zeros predictor on alternating labels")


@check("on pure noise the leaky pipeline scores far above chance and the fold-safe pipeline scores near 0.5")
def test_planted_leak(sub):
    leaky_fn = has(sub, "leaky_cv_accuracy")
    safe_fn = has(sub, "safe_cv_accuracy")
    leaky, safe = [], []
    for seed in range(10):
        X, y = noise_problem(seed)
        folds = [np.sort(f) for f in np.array_split(np.random.default_rng(100 + seed).permutation(50), 5)]
        leaky.append(leaky_fn(X, y, 10, folds))
        safe.append(safe_fn(X, y, 10, folds))
    leaky_mean, safe_mean = float(np.mean(leaky)), float(np.mean(safe))
    if leaky_mean < 0.75:
        fail(f"leaky_cv_accuracy averaged {leaky_mean:.3f} over 10 noise datasets, expected above 0.75; "
             f"select the columns on all rows before cross-validating")
    # Wrong implementation caught here: a safe_cv_accuracy that still calls select_top_k(X, y, ...) on all
    # rows before the folds. It averages about 0.85 on noise, not 0.5.
    if not 0.42 <= safe_mean <= 0.58:
        fail(f"safe_cv_accuracy averaged {safe_mean:.3f} on pure noise, expected within 0.42 to 0.58; "
             f"select the columns inside each fold with the training rows only")


@check("count_cross_fold_duplicates counts held-out rows that have an exact twin in the training rows")
def test_duplicates(sub):
    fn = has(sub, "count_cross_fold_duplicates")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    folds = [np.arange(0, 5), np.arange(5, 10)]
    equal(fn(X, folds), 0, label="duplicates among 10 distinct rows")
    X2 = np.vstack([X, X[[0, 7]]])  # rows 10 and 11 copy rows 0 and 7
    folds2 = [np.array([0, 1, 2, 3, 4, 11]), np.array([5, 6, 7, 8, 9, 10])]
    equal(fn(X2, folds2), 4,
          label="two twin pairs split across the folds, each pair counted once per round, so 4")
