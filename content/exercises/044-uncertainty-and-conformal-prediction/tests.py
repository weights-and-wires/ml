"""Checks for the uncertainty and conformal prediction exercise."""

import math

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.1, 1.5])


def _random_feature_model(x, y, seed, width=60, lam=1e-2):
    """Ridge regression on tanh random features; the seed picks the features."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=1.5, size=width)
    b = rng.uniform(-2.0, 2.0, size=width)

    def feats(t):
        return np.tanh(np.outer(np.asarray(t, dtype=float), w) + b)

    phi = feats(x)
    beta = np.linalg.solve(phi.T @ phi + lam * np.eye(width), phi.T @ y)
    return lambda t: feats(t) @ beta


def _constant(values):
    values = np.asarray(values, dtype=float)
    return lambda X: values


@check("ensemble_predict returns the mean and variance over models: sizes 1 and 4 give 0.50 and 1.00, and 0.0003 and 0.107")
def test_ensemble_predict_hand(sub):
    ensemble_predict = has(sub, "ensemble_predict")
    models = [_constant([0.50, 1.00]), _constant([0.52, 1.40]), _constant([0.48, 0.60])]
    mean, var = ensemble_predict(models, np.array([1.0, 4.0]))
    shape(mean, (2,), label="mean shape")
    shape(var, (2,), label="variance shape")
    close(mean, np.array([0.50, 1.00]), atol=1e-12, label="ensemble mean")
    close(var, np.array([0.0004 * 2 / 3, 0.32 / 3]), atol=1e-12, label="ensemble variance (divide by M, reduce over models)")


@check("an ensemble of seeded random-feature models disagrees far more outside the training range than inside it")
def test_ensemble_spread_grows(sub):
    ensemble_predict = has(sub, "ensemble_predict")
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, size=30)
    y = np.sin(2.0 * x) + 0.15 * rng.normal(size=30)
    models = [_random_feature_model(x, y, seed) for seed in range(8)]
    inside = np.linspace(-2.0, 2.0, 81)
    outside = np.concatenate([np.linspace(-4.0, -2.5, 20), np.linspace(2.5, 4.0, 20)])
    _, var_in = ensemble_predict(models, inside)
    _, var_out = ensemble_predict(models, outside)
    shape(var_in, inside.shape, label="variance shape inside")
    if not var_out.mean() > 20.0 * var_in.mean():
        fail(f"mean variance outside {var_out.mean():.5f} should exceed 20 times the mean variance inside {var_in.mean():.5f}")
    mean_in, _ = ensemble_predict(models, inside)
    almost(float(np.mean((mean_in - np.sin(2.0 * inside)) ** 2)), 0.0, tol=0.05, label="mean squared error of the ensemble mean inside the range")


@check("mc_dropout_predict calls forward n times with the same rng and reduces over the passes")
def test_mc_dropout_calls(sub):
    mc_dropout_predict = has(sub, "mc_dropout_predict")
    seen = []

    def forward(X, p, rng):
        seen.append((p, id(rng)))
        return X + rng.normal(size=X.shape)

    rng = np.random.default_rng(3)
    X = np.array([0.0, 1.0, 2.0])
    mean, var = mc_dropout_predict(forward, X, 0.25, 7, rng)
    if len(seen) != 7:
        fail(f"forward should be called 7 times, it was called {len(seen)} times")
    if any(p != 0.25 or r != id(rng) for p, r in seen):
        fail("forward must receive p = 0.25 and the same rng object on every call")
    rng2 = np.random.default_rng(3)
    passes = np.stack([X + rng2.normal(size=X.shape) for _ in range(7)])
    close(mean, passes.mean(axis=0), atol=1e-12, label="mean over the 7 passes")
    close(var, passes.var(axis=0), atol=1e-12, label="variance over the 7 passes")


@check("MC dropout on four units contributing [1, 2, 3, 4] at p = 0.5 has mean 10 x and variance 30 x^2, and variance 0 at p = 0")
def test_mc_dropout_exact(sub):
    mc_dropout_predict = has(sub, "mc_dropout_predict")
    c = np.array([1.0, 2.0, 3.0, 4.0])

    def forward(X, p, rng):
        mask = (rng.random((len(X), 4)) < 1.0 - p).astype(float)
        return X * (mask * c / (1.0 - p)).sum(axis=1)

    X = np.array([1.0, 2.0])
    mean, var = mc_dropout_predict(forward, X, 0.5, 4000, np.random.default_rng(0))
    close(mean, 10.0 * X, rtol=0.05, label="MC mean, near [10, 20]")
    close(var, 30.0 * X**2, rtol=0.12, label="MC variance, near [30, 120]")
    _, var0 = mc_dropout_predict(forward, X, 0.0, 50, np.random.default_rng(0))
    close(var0, np.zeros(2), atol=1e-12, label="variance with no dropout")


# Plausible wrong implementation caught by the coverage check below (and by the
# infinite-quantile case here): the quantile at level 1 - alpha without the
# (n + 1) correction, k = ceil(n (1 - alpha)) or np.quantile(scores, 1 - alpha).
# At n = 10 and alpha = 0.1 it covers about 9/11 = 0.82 instead of 10/11 = 0.91.
@check("conformal_quantile is the ceil((n + 1)(1 - alpha))-th smallest score, and infinite when that index passes n")
def test_conformal_quantile(sub):
    conformal_quantile = has(sub, "conformal_quantile")
    almost(float(conformal_quantile(SCORES, 0.2)), 1.1, tol=1e-12, label="nine scores, alpha = 0.2 (k = 8)")
    almost(float(conformal_quantile(SCORES[::-1].copy(), 0.1)), 1.5, tol=1e-12, label="nine scores in reverse order, alpha = 0.1 (k = 9)")
    got = float(conformal_quantile(np.array([0.3, 0.1, 0.2]), 0.1))
    if not math.isinf(got):
        fail(f"three scores at alpha = 0.1 need k = 4 > n = 3, expected inf, got {got}")


@check("conformal_interval puts q = 1.1 on each side of a prediction of 3.0, giving [1.9, 4.1]")
def test_conformal_interval_hand(sub):
    conformal_interval = has(sub, "conformal_interval")
    y_cal = np.array([1.5, -0.3, 0.7, -0.5, 0.2, 1.1, -0.1, 0.8, -0.4])
    pred_cal = np.zeros(9)
    # |y_cal| is SCORES in a shuffled order, with mixed signs.
    lo, hi = conformal_interval(pred_cal, y_cal, np.array([3.0, 0.0]), 0.2)
    shape(lo, (2,), label="lower shape")
    close(lo, np.array([1.9, -1.1]), atol=1e-12, label="lower ends")
    close(hi, np.array([4.1, 1.1]), atol=1e-12, label="upper ends")


@check("over 200 seeded splits with n = 10 and alpha = 0.1, interval coverage averages near 10/11, not the 9/11 of the uncorrected quantile")
def test_interval_coverage(sub):
    conformal_interval = has(sub, "conformal_interval")
    rng = np.random.default_rng(7)
    covs = []
    for _ in range(200):
        x_cal = rng.uniform(-2, 2, 10)
        y_cal = np.sin(x_cal) + 0.3 * rng.standard_t(5, 10)
        x_te = rng.uniform(-2, 2, 500)
        y_te = np.sin(x_te) + 0.3 * rng.standard_t(5, 500)
        lo, hi = conformal_interval(np.sin(x_cal), y_cal, np.sin(x_te), 0.1)
        covs.append(np.mean((y_te >= lo) & (y_te <= hi)))
    almost(float(np.mean(covs)), 10.0 / 11.0, tol=0.03, label="mean coverage over 200 splits")


@check("conformal_sets keeps the classes with probability at least 0.4 on the nine-example case, and averages near 90 percent coverage")
def test_conformal_sets(sub):
    conformal_sets = has(sub, "conformal_sets")
    true_p = np.array([0.9, 0.8, 0.95, 0.6, 0.7, 0.5, 0.85, 0.4, 0.99])
    probs_cal = np.zeros((9, 3))
    probs_cal[:, 0] = true_p
    probs_cal[:, 1] = 1.0 - true_p
    y_cal = np.zeros(9, dtype=int)
    probs_test = np.array([[0.7, 0.2, 0.1], [0.45, 0.40, 0.15], [0.3, 0.3, 0.4]])
    got = np.asarray(conformal_sets(probs_cal, y_cal, probs_test, 0.1))
    want = np.array([[True, False, False], [True, True, False], [False, False, True]])
    shape(got, (3, 3), label="set shape")
    if not np.array_equal(got, want):
        fail(f"expected sets\n{want}\ngot\n{got}")
    rng = np.random.default_rng(11)
    covered = []
    sizes = []
    for _ in range(200):
        probs = rng.dirichlet(np.full(4, 0.3), size=250)
        labels = np.array([rng.choice(4, p=row) for row in probs])
        sets = np.asarray(conformal_sets(probs[:50], labels[:50], probs[50:], 0.1))
        covered.append(sets[np.arange(200), labels[50:]].mean())
        sizes.append(sets.sum(axis=1).mean())
    almost(float(np.mean(covered)), 0.9, tol=0.03, label="mean coverage over 200 splits")
    if not np.mean(sizes) < 4.0:
        fail(f"sets should be smaller than all 4 classes on average, got {np.mean(sizes):.2f}")
