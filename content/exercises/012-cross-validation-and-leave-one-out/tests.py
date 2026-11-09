"""Checks for the cross-validation and leave-one-out exercise."""

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


def brute_force_loo(X, y, lam):
    n = X.shape[0]
    errors = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        w = ridge_fit(X[keep], y[keep], lam)
        errors[i] = (y[i] - X[i] @ w) ** 2
    return float(np.mean(errors))


def loo_data(n=50, d=4, seed=1, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + noise * rng.normal(size=n)
    return X, y


HAND_X = np.array([[1.0], [2.0], [3.0]])
HAND_Y = np.array([1.0, 2.0, 2.0])


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


@check("hat_matrix is n by n, symmetric, gives the ridge fitted values, and is a projection at lam=0")
def test_hat_matrix(sub):
    got = has(sub, "hat_matrix")
    X, y = loo_data()
    for lam in (0.0, 2.0):
        H = got(X, lam)
        shape(H, (50, 50), label=f"hat matrix at lam={lam}")
        close(H, H.T, atol=1e-9, label="H should be symmetric")
        close(H @ y, X @ ridge_fit(X, y, lam), atol=1e-8,
              label=f"H @ y against the ridge fitted values at lam={lam}")
    X2, _ = loo_data(n=30, d=3, seed=2)
    H0 = got(X2, 0.0)
    close(H0 @ H0, H0, atol=1e-8, label="H @ H against H at lam=0")


@check("leverages of the hand example are 1/15, 4/15, 9/15, and in general lie in [0, 1) and sum to the degrees of freedom")
def test_leverages_hand(sub):
    got = has(sub, "leverages")
    close(got(HAND_X, 1.0), np.array([1, 4, 9]) / 15.0, atol=1e-12,
          label="leverages for inputs 1, 2, 3 at lam=1")
    X, _ = loo_data(n=40, d=6, seed=3)
    s = np.linalg.svd(X, compute_uv=False)
    for lam in (0.1, 1.0, 20.0):
        h = got(X, lam)
        shape(h, (40,), label="leverages")
        if np.any(h < 0) or np.any(h >= 1):
            fail(f"at lam={lam} every leverage should lie in [0, 1), got min {h.min():.4g}, max {h.max():.4g}")
        close(h.sum(), np.sum(s ** 2 / (s ** 2 + lam)), atol=1e-8,
              label=f"sum of leverages at lam={lam}")


@check("loo_linear_smoother returns 0.2869 on the hand example and matches n separate refits")
def test_loo_hand(sub):
    got = has(sub, "loo_linear_smoother")
    want = (4 / 49 + 64 / 121 + 1 / 4) / 3
    close(got(HAND_X, HAND_Y, 1.0), want, atol=1e-10, label="leave-one-out MSE for inputs 1, 2, 3")
    X, y = loo_data(n=50, d=4, seed=1)
    for lam in (0.0, 0.5, 5.0):
        close(got(X, y, lam), brute_force_loo(X, y, lam), rtol=1e-6, atol=1e-9,
              label=f"leave-one-out MSE at lam={lam}")
