"""Checks for the cross-validation exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def ridge_predict(w, X):
    return X @ w


def houses(seed=7, n=60, d=30):
    """Sixty houses with thirty overlapping measurements, as in the chapter."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = Z @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = 0.3 * rng.normal(size=d)
    y = X @ w_true + rng.normal(size=n)
    return X, y, w_true, loadings


@check("k_fold_indices cuts the rows into k disjoint folds that cover every row once")
def test_partition(sub):
    got = has(sub, "k_fold_indices")
    n, k = 23, 5
    folds = got(n, k, np.random.default_rng(3))
    if len(folds) != k:
        fail(f"expected {k} folds, got {len(folds)}")
    whole = np.sort(np.concatenate(folds))
    if not np.array_equal(whole, np.arange(n)):
        fail(f"the folds should hold each of 0..{n - 1} exactly once, got {whole.tolist()}")
    for a in range(k):
        if not np.array_equal(folds[a], np.sort(folds[a])):
            fail(f"fold {a} should be sorted, got {folds[a].tolist()}")


@check("k_fold_indices makes sizes 3, 2, 2 for 7 rows in 3 folds, and shuffles with the generator")
def test_sizes_and_shuffle(sub):
    got = has(sub, "k_fold_indices")
    folds = got(7, 3, np.random.default_rng(0))
    sizes = sorted(len(f) for f in folds)
    if sizes != [2, 2, 3]:
        fail(f"7 rows in 3 folds should give sizes 3, 2, 2, got {sizes}")
    a = got(30, 5, np.random.default_rng(1))
    b = got(30, 5, np.random.default_rng(1))
    c = got(30, 5, np.random.default_rng(2))
    if any(not np.array_equal(x, y) for x, y in zip(a, b)):
        fail("the same generator seed should give the same folds")
    if all(np.array_equal(x, y) for x, y in zip(a, c)):
        fail("different generator seeds should give different folds; did you shuffle?")


@check("cross_val_score returns k finite errors and the same errors on every call")
def test_shape_and_determinism(sub):
    got = has(sub, "cross_val_score")
    X, y, _, _ = houses()
    fit = lambda Xt, yt: ridge_fit(Xt, yt, 1.0)
    first = got(fit, ridge_predict, X, y, k=4)
    shape(first, (4,), label="cross_val_score output")
    if not np.all(np.isfinite(first)) or np.any(first < 0):
        fail(f"held-out squared errors must be finite and nonnegative, got {first}")
    close(first, got(fit, ridge_predict, X, y, k=4), atol=1e-12,
          label="a second call should reuse the same folds")


@check("cross_val_score scores every row exactly once: a model that predicts 0 gets the mean of y squared")
def test_every_row_once(sub):
    got = has(sub, "cross_val_score")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    scores = got(lambda Xt, yt: None, lambda m, Xs: np.zeros(len(Xs)), X, y, k=4)
    close(scores.mean(), np.mean(y ** 2), rtol=1e-9,
          label="average of the 4 fold scores for 40 rows, 10 per fold")


@check("cross_val_score never lets a held-out row into the training set")
def test_no_leak(sub):
    got = has(sub, "cross_val_score")
    rng = np.random.default_rng(6)
    n = 30
    X = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    y = rng.normal(size=n)

    def fit(Xt, yt):
        return set(Xt[:, 0].astype(int).tolist())

    def predict(seen, Xs):
        test_ids = set(Xs[:, 0].astype(int).tolist())
        if seen & test_ids:
            fail(f"rows {sorted(seen & test_ids)} were in both the training and the held-out set")
        return np.zeros(len(Xs))

    got(fit, predict, X, y, k=5)


@check("cross_val_score gives about zero error when a linear model fits noise-free data")
def test_noise_free(sub):
    got = has(sub, "cross_val_score")
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0])
    scores = got(lambda Xt, yt: ridge_fit(Xt, yt, 0.0), ridge_predict, X, y, k=5)
    close(scores, np.zeros(5), atol=1e-12, label="held-out errors of an exact linear fit")


@check("best_lambda_by_cv returns one mean per penalty and the penalty with the smallest mean")
def test_best_lambda_consistent(sub):
    best_fn = has(sub, "best_lambda_by_cv")
    score_fn = has(sub, "cross_val_score")
    X, y, _, _ = houses()
    lambdas = np.logspace(-2, 3, 8)
    best, means = best_fn(X, y, lambdas, k=5)
    shape(np.asarray(means), (8,), label="cv_means")
    want = np.array([
        score_fn(lambda Xt, yt, l=l: ridge_fit(Xt, yt, l), ridge_predict, X, y, k=5).mean()
        for l in lambdas
    ])
    close(means, want, rtol=1e-8, label="mean fold error of each penalty")
    close(best, lambdas[int(np.argmin(want))], label="the chosen penalty")


@check("the penalty chosen by cross-validation beats no penalty and a huge penalty on fresh houses")
def test_chosen_penalty_generalises(sub):
    best_fn = has(sub, "best_lambda_by_cv")
    X, y, w_true, loadings = houses()
    lambdas = np.logspace(-2, 3, 20)
    best, _ = best_fn(X, y, lambdas, k=5)

    rng = np.random.default_rng(99)
    X_new = rng.normal(size=(5000, 5)) @ loadings + 0.5 * rng.normal(size=(5000, X.shape[1]))
    y_new = X_new @ w_true + rng.normal(size=5000)

    def fresh_error(lam):
        return float(np.mean((y_new - X_new @ ridge_fit(X, y, lam)) ** 2))

    chosen, tiny, huge = fresh_error(best), fresh_error(0.01), fresh_error(1000.0)
    if chosen >= tiny:
        fail(f"penalty {best:.3g} gives fresh error {chosen:.3f}, no better than lambda 0.01 at {tiny:.3f}")
    if chosen >= huge:
        fail(f"penalty {best:.3g} gives fresh error {chosen:.3f}, no better than lambda 1000 at {huge:.3f}")
