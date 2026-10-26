"""Checks for the house-pricing project."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite, finite_difference, has, shape

NOISE = slice(6, 11)


def _ridge_direct(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


@check("split_indices makes three disjoint, seeded groups that cover every row")
def test_split(sub):
    split = has(sub, "split_indices")
    tr, va, te = split(500, 3)
    equal(len(tr), 300, label="training rows")
    equal(len(va), 100, label="validation rows")
    equal(len(te), 100, label="test rows")
    every = np.sort(np.concatenate([tr, va, te]))
    equal(every, np.arange(500), label="rows used across the three groups")
    tr2, va2, te2 = split(500, 3)
    equal(np.concatenate([tr, va, te]), np.concatenate([tr2, va2, te2]), label="same seed")
    if np.array_equal(tr, split(500, 4)[0]):
        fail("seeds 3 and 4 gave the same training rows; the shuffle ignores the seed")
    if np.array_equal(np.sort(te), np.arange(400, 500)) or np.array_equal(np.sort(tr), np.arange(300)):
        fail("the groups are consecutive blocks, so the rows were not shuffled")


@check("The scaler learns from training rows only and standardises with them")
def test_scaler(sub):
    rng = np.random.default_rng(0)
    X_tr = rng.normal([10.0, -3.0, 0.0], [5.0, 0.1, 1.0], size=(200, 3))
    X_val = rng.normal([20.0, -3.0, 0.0], [5.0, 0.1, 1.0], size=(50, 3))
    mean, std = sub.fit_scaler(X_tr)
    close(mean, X_tr.mean(axis=0), label="mean")
    close(std, X_tr.std(axis=0), label="std (divide by n)")
    Z_tr = sub.apply_scaler(X_tr, mean, std)
    close(Z_tr.mean(axis=0), np.zeros(3), atol=1e-10, label="training column means")
    close(Z_tr.std(axis=0), np.ones(3), atol=1e-10, label="training column stds")
    Z_val = sub.apply_scaler(X_val, mean, std)
    if abs(Z_val[:, 0].mean()) < 1.0:
        fail(
            f"validation column 0 has mean {Z_val[:, 0].mean():.2f} after scaling; it should stay "
            "near +2 because the shift of 10 units is measured in training standard deviations"
        )
    _, std_const = sub.fit_scaler(np.column_stack([np.ones(10), np.arange(10.0)]))
    finite(sub.apply_scaler(np.ones((3, 2)), np.array([1.0, 0.0]), std_const), label="constant column")


@check("fit_qr stays accurate when two columns are nearly identical")
def test_qr(sub):
    rng = np.random.default_rng(1)
    a = rng.normal(size=100)
    X = np.column_stack([a, a + 1e-7 * rng.normal(size=100), rng.normal(size=100)])
    w_true = np.array([1.0, 2.0, -1.0])
    y = X @ w_true
    got = sub.fit_qr(X, y)
    shape(got, (3,))
    close(got, w_true, atol=1e-4, label="weights on a condition number near 1e7 (the normal equations lose these digits)")
    X2 = rng.normal(size=(60, 4))
    y2 = rng.normal(size=60)
    close(sub.fit_qr(X2, y2), np.linalg.lstsq(X2, y2, rcond=None)[0], atol=1e-9, label="random problem")


@check("ridge_path returns one ridge solution per lambda from a single SVD")
def test_ridge_path(sub):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 6))
    X[:, 5] = X[:, 4] + 0.01 * rng.normal(size=80)
    y = rng.normal(size=80)
    lams = np.array([0.0, 0.01, 1.0, 50.0, 1e6])
    W = sub.ridge_path(X, y, lams)
    shape(W, (5, 6))
    for k, lam in enumerate(lams[1:], start=1):
        close(W[k], _ridge_direct(X, y, lam), rtol=1e-6, atol=1e-8, label=f"lambda = {lam:g}")
    close(W[0], np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-6, atol=1e-6, label="lambda = 0 is least squares")
    if np.abs(W[-1]).max() > 1e-3:
        fail(f"lambda = 1e6 should push every weight near 0, but the largest is {np.abs(W[-1]).max():.3g}")


@check("choose_lambda picks the fee with the lowest validation error")
def test_choose_lambda(sub):
    rng = np.random.default_rng(3)
    w_true = rng.normal(size=20)
    X_tr, X_val = rng.normal(size=(30, 20)), rng.normal(size=(300, 20))
    y_tr = X_tr @ w_true + 2.0 * rng.normal(size=30)
    y_val = X_val @ w_true + 2.0 * rng.normal(size=300)
    lams = np.logspace(-2, 3, 21)
    best, errors, w = sub.choose_lambda(X_tr, y_tr, X_val, y_val, lams)
    shape(errors, (21,))
    want = np.array([np.sqrt(np.mean((X_val @ _ridge_direct(X_tr, y_tr, l) - y_val) ** 2)) for l in lams])
    close(errors, want, rtol=1e-5, label="validation RMSE per lambda (fit on training rows)")
    almost(best, lams[int(np.argmin(want))], 1e-9, label="best lambda")
    close(w, _ridge_direct(X_tr, y_tr, best), rtol=1e-5, atol=1e-8, label="weights for the best lambda")
    if best <= lams[0] or best >= lams[-1]:
        fail(f"expected an interior best lambda on this problem, got the grid end {best:g}")


@check("lambda_max zeroes every lasso weight, and a smaller fee lets one in")
def test_lambda_max(sub):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 8))
    y = X @ np.array([3.0, -2.0, 0, 0, 1.0, 0, 0, 0]) + rng.normal(size=120)
    lm = sub.lambda_max(X, y)
    almost(lm, np.abs(X.T @ y).max(), 1e-9, label="lambda_max")
    w_hi = sub.fista_lasso(X, y, 1.001 * lm, iters=100)
    if np.any(w_hi != 0):
        fail(f"at lambda just above lambda_max every weight should be exactly 0, got {np.round(w_hi, 4)}")
    w_lo = sub.fista_lasso(X, y, 0.5 * lm, iters=300)
    if not np.any(w_lo != 0):
        fail("at half of lambda_max at least one weight should be non-zero, but all are 0")


@check("fista_lasso satisfies the lasso optimality (KKT) conditions with exact zeros")
def test_fista(sub):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 12))
    X[:, 11] = X[:, 10] + 0.05 * rng.normal(size=200)
    w_true = np.zeros(12)
    w_true[[0, 3, 10]] = [4.0, -3.0, 2.0]
    y = X @ w_true + rng.normal(size=200)
    lam = 0.2 * np.abs(X.T @ y).max()
    w = sub.fista_lasso(X, y, lam, iters=600)
    shape(w, (12,))
    g = X.T @ (X @ w - y)
    on = w != 0
    if np.any(on) and np.abs(g[on] + lam * np.sign(w[on])).max() > 1e-3 * lam:
        fail(f"non-zero weights need gradient = -lam * sign(w); worst miss {np.abs(g[on] + lam * np.sign(w[on])).max():.3g}")
    if np.any(~on) and (np.abs(g[~on]) - lam).max() > 1e-3 * lam:
        fail(f"zero weights need |gradient| <= lam = {lam:.3g}; worst is {np.abs(g[~on]).max():.3g}")
    if int(np.sum(~on)) < 5:
        fail(f"expected most of the 9 irrelevant columns to be exactly 0, got {int(np.sum(~on))} zeros in total")


@check("add_intercept and mean_log_loss behave on known values")
def test_small_pieces(sub):
    A = sub.add_intercept(np.array([[2.0, 3.0], [4.0, 5.0]]))
    close(A, np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]), label="add_intercept")
    p, y = np.array([0.9, 0.2]), np.array([1.0, 0.0])
    close(sub.mean_log_loss(p, y), -(np.log(0.9) + np.log(0.8)) / 2, label="mean_log_loss")
    got = sub.mean_log_loss(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    if not np.isfinite(got) or got < 10:
        fail(f"a confident wrong answer should cost a large finite number, got {got}")


@check("fit_logistic_newton reaches a stationary point of the penalised loss")
def test_newton(sub):
    rng = np.random.default_rng(6)
    X = np.column_stack([np.ones(300), rng.normal(size=(300, 4))])
    X[:, 4] = X[:, 3] + 0.001 * rng.normal(size=300)
    y = (rng.uniform(size=300) < sub.sigmoid(X @ np.array([0.3, 1.0, -1.0, 0.8, 0.0]))).astype(float)
    ridge = 1.0
    w = sub.fit_logistic_newton(X, y, ridge=ridge)
    shape(w, (5,))
    g = finite_difference(lambda v: sub.logistic_loss(v, X, y, ridge), w.copy(), eps=1e-5)
    if np.linalg.norm(g) > 1e-4:
        fail(f"finite-difference gradient of the penalised loss at your weights has norm {np.linalg.norm(g):.3g}, expected under 1e-4")
    Xs = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    ys = np.array([0.0, 0.0, 1.0, 1.0])
    finite(sub.fit_logistic_newton(Xs, ys, ridge=1.0), label="separable data with ridge")


@check("The pipeline gives an honest test error close to the price noise")
def test_pipeline_regression(sub):
    for seed in range(5):
        r = sub.run_pipeline(seed)
        for key in ("rmse_ols", "rmse_ridge", "rmse_lasso", "lambda", "w_ols", "w_ridge", "w_lasso"):
            if key not in r:
                fail(f"run_pipeline({seed}) is missing the key {key!r}")
        if not r["rmse_ridge"] < 33.0:
            fail(f"seed {seed}: ridge test RMSE {r['rmse_ridge']:.1f}, expected under 33 (the noise in price is 25)")
        if not r["rmse_ridge"] > 15.0:
            fail(f"seed {seed}: test RMSE {r['rmse_ridge']:.1f} is below the noise floor of 25; test rows leaked into fitting")
        if not r["rmse_ols"] < 36.0:
            fail(f"seed {seed}: QR least squares test RMSE {r['rmse_ols']:.1f}, expected under 36")
        if not np.any(np.isclose(sub.LAMBDAS, r["lambda"])):
            fail(f"seed {seed}: chosen lambda {r['lambda']} is not on the LAMBDAS grid")


@check("The lasso zeroes the noise columns and keeps the planted ones")
def test_pipeline_lasso(sub):
    for seed in range(5):
        w = np.asarray(sub.run_pipeline(seed)["w_lasso"])
        shape(w, (11,))
        if np.any(w[NOISE] != 0):
            fail(f"seed {seed}: noise columns got non-zero weights {np.round(w[NOISE], 3)}; they should be exactly 0")
        if w[0] == 0 and w[1] == 0:
            fail(f"seed {seed}: both area columns are 0, but area drives the price")
        for idx, name in ((3, "age"), (4, "age_gap_sq"), (5, "dist_km")):
            if w[idx] == 0:
                fail(f"seed {seed}: the planted column {name} was set to 0")
        if w[5] >= 0:
            fail(f"seed {seed}: distance should push the price down, but its weight is {w[5]:.2f}")


@check("The classifier beats the always-say-the-base-rate predictor on test log loss")
def test_pipeline_classifier(sub):
    for seed in range(5):
        r = sub.run_pipeline(seed)
        gain = r["base_log_loss"] - r["log_loss"]
        if not np.isfinite(r["log_loss"]) or gain < 0.08:
            fail(
                f"seed {seed}: test log loss {r['log_loss']:.3f} vs base-rate {r['base_log_loss']:.3f}; "
                f"expected a gain of at least 0.08, got {gain:.3f}"
            )
