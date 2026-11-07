"""Checks for the learning-curve and winner's-curse exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def ridge_fit(X, y, lam=1e-6):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def ridge_predict(w, X):
    return X @ w


def linear_data(n=300, d=5, seed=4, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return X, X @ rng.normal(size=d) + noise * rng.normal(size=n)


@check("validation_split sets aside round(val_fraction * n) rows, covers every row once, and repeats for one seed")
def test_split_sizes(sub):
    got = has(sub, "validation_split")
    val, pool = got(23, 0.2, 0)
    equal(len(val), 5, label="validation rows for n=23, fraction 0.2")
    equal(len(pool), 18, label="pool rows for n=23, fraction 0.2")
    whole = np.sort(np.concatenate([val, pool]))
    if not np.array_equal(whole, np.arange(23)):
        fail(f"the validation and pool rows should cover 0..22 exactly once, got {whole.tolist()}")
    a, pa = got(50, 0.3, 11)
    b, pb = got(50, 0.3, 11)
    c, pc = got(50, 0.3, 12)
    if not (np.array_equal(a, b) and np.array_equal(pa, pb)):
        fail("the same seed should give the same split")
    if np.array_equal(pa, pc):
        fail("a different seed should shuffle the pool differently")


@check("learning_curve returns one finite training error and one validation error per size")
def test_curve_shapes(sub):
    got = has(sub, "learning_curve")
    X, y = linear_data()
    train, val = got(ridge_fit, ridge_predict, X, y, [10, 40, 160])
    shape(train, (3,), label="training errors")
    shape(val, (3,), label="validation errors")
    if not (np.all(np.isfinite(train)) and np.all(np.isfinite(val))):
        fail(f"errors must be finite, got {train} and {val}")


@check("learning_curve trains on nested prefixes of the pool and scores one fixed validation set")
def test_nesting(sub):
    got = has(sub, "learning_curve")
    n = 80
    rng = np.random.default_rng(2)
    X = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    y = rng.normal(size=n)
    train_sets, val_sets = [], []

    def fit(Xt, yt):
        train_sets.append(set(Xt[:, 0].astype(int).tolist()))
        return None

    def predict(model, Xs):
        ids = set(Xs[:, 0].astype(int).tolist())
        if ids <= train_sets[-1]:
            return np.zeros(len(Xs))  # scoring the training rows themselves
        if ids & train_sets[-1]:
            fail("a scored set mixes training rows with other rows")
        val_sets.append(ids)
        return np.zeros(len(Xs))

    got(fit, predict, X, y, [5, 12, 30], val_fraction=0.25, seed=0)
    for small, large in zip(train_sets, train_sets[1:]):
        if not small <= large:
            fail("each training set should contain the previous one, but a smaller set has rows the larger one lacks")
    if [len(s) for s in train_sets] != [5, 12, 30]:
        fail(f"training set sizes should be 5, 12, 30, got {[len(s) for s in train_sets]}")
    if len(val_sets) != 3 or any(v != val_sets[0] for v in val_sets):
        fail("every size should be scored on the same validation rows")
    if val_sets[0] & train_sets[-1]:
        fail("validation rows must never appear in a training set")
    equal(len(val_sets[0]), 20, label="validation rows for n=80, fraction 0.25")


@check("on linear data the training error starts near 0 and the validation error falls with more data")
def test_curve_behaviour(sub):
    got = has(sub, "learning_curve")
    X, y = linear_data(n=300, d=5, seed=4, noise=1.0)
    train, val = got(ridge_fit, ridge_predict, X, y, [5, 40, 160], val_fraction=0.25, seed=1)
    if train[0] > 1e-6:
        fail(f"5 points and 5 weights can be fitted exactly, so the training error should be near 0, got {train[0]:.3g}")
    if not (val[0] > val[1] > val[2] * 0.95):
        fail(f"validation error should fall as the training set grows, got {np.round(val, 3).tolist()}")
    if not train[0] < train[2]:
        fail(f"training error should rise as the training set grows, got {np.round(train, 3).tolist()}")


@check("diagnose_curve names the three verdicts on hand-built curves")
def test_diagnose_hand(sub):
    got = has(sub, "diagnose_curve")
    cases = [
        ("underfit", [0.9, 0.95, 1.0], [1.4, 1.1, 1.05]),
        ("overfit", [0.0, 0.05, 0.1], [3.0, 1.5, 0.9]),
        ("good", [0.2, 0.15, 0.15], [0.5, 0.25, 0.2]),
        ("overfit", [0.1, 0.3, 0.5], [2.0, 1.6, 1.3]),
    ]
    for want, train, val in cases:
        verdict = got(np.array(train), np.array(val), 0.5, 0.2)
        if verdict != want:
            fail(f"train {train}, validation {val} should be {want!r}, got {verdict!r}")


@check("diagnose_curve calls a straight line on curved prices 'underfit' and a degree 8 curve 'overfit'")
def test_diagnose_real_curves(sub):
    lc, diag = has(sub, "learning_curve"), has(sub, "diagnose_curve")
    rng = np.random.default_rng(3)
    n = 60
    area = rng.uniform(-1, 1, n)
    price = np.sin(3 * area) + 0.5 * rng.normal(size=n)
    sizes = [10, 14, 18, 24, 30, 38, 45]

    def run(degree):
        def fit(x, y):
            F = np.vander(x[:, 0], degree + 1, increasing=True)
            return ridge_fit(F, y, 1e-3)

        def predict(w, x):
            return np.vander(x[:, 0], degree + 1, increasing=True) @ w

        runs = [lc(fit, predict, area[:, None], price, sizes, 0.25, seed=s) for s in range(60)]
        return (np.median([r[0] for r in runs], axis=0), np.median([r[1] for r in runs], axis=0))

    for degree, want in ((1, "underfit"), (8, "overfit")):
        train, val = run(degree)
        verdict = diag(train, val, 0.4, 0.08)
        if verdict != want:
            fail(f"degree {degree}: final training error {train[-1]:.3f}, validation {val[-1]:.3f}; "
                 f"expected {want!r}, got {verdict!r}")


@check("best_of_m_gap is 0 for one model, about 0.187 for 20, grows with m, scales with the noise, and ignores the true error")
def test_gap_values(sub):
    got = has(sub, "best_of_m_gap")
    one = got(1.0, 0.1, 1, 20000, np.random.default_rng(0))
    close(one, 0.0, atol=0.005, label="best of 1 model")
    twenty = got(1.0, 0.1, 20, 20000, np.random.default_rng(0))
    close(twenty, 0.187, atol=0.01, label="best of 20 models (1.87 noise standard deviations)")
    gaps = [got(1.0, 0.1, m, 20000, np.random.default_rng(1)) for m in (2, 5, 20, 100)]
    if not all(a < b for a, b in zip(gaps, gaps[1:])):
        fail(f"the gap should grow with m for m = 2, 5, 20, 100, got {np.round(gaps, 3).tolist()}")
    small = got(1.0, 0.1, 20, 20000, np.random.default_rng(2))
    large = got(1.0, 0.2, 20, 20000, np.random.default_rng(2))
    close(large / small, 2.0, rtol=1e-6, label="doubling the noise should double the gap")
    shifted = got(5.0, 0.1, 20, 20000, np.random.default_rng(2))
    close(shifted, small, atol=1e-9, label="the gap should not depend on the true error")


@check("empirical_rademacher is near 0 for one model, grows for 300, gives 0.5 on the two-model example, and 1 for all 128 sign patterns")
def test_rademacher_grows(sub):
    got = has(sub, "empirical_rademacher")
    rng = np.random.default_rng(2)
    one = rng.choice(np.array([-1.0, 1.0]), size=(1, 40))
    small = got(one, np.random.default_rng(10), 4000)
    if abs(small) > 0.05:
        fail(f"one model averages about 0 against random signs, got {small:.3g}")
    many = rng.choice(np.array([-1.0, 1.0]), size=(300, 40))
    large = got(many, np.random.default_rng(11), 4000)
    if not 0.35 < large <= 1.0:
        fail(f"300 random sign models on 40 examples should score about 0.45, got {large:.3g}")
    two = np.array([[1.0, 1.0], [1.0, -1.0]])
    close(got(two, np.random.default_rng(3), 20000), 0.5, atol=0.01, label="models [1, 1] and [1, -1]")
    n = 7
    patterns = np.array([[1.0 if (i >> b) & 1 else -1.0 for b in range(n)] for i in range(2 ** n)])
    close(got(patterns, np.random.default_rng(12), 3000), 1.0, atol=1e-9,
          label="all 128 sign patterns on 7 examples")


@check("nested_cv_score with one candidate equals plain cross-validation, and finds the exact penalty on noise-free data")
def test_nested_single(sub):
    got = has(sub, "nested_cv_score")
    cv = has(sub, "cross_val_score")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 6))
    y = X @ rng.normal(size=6) + rng.normal(size=50)
    want = cv(lambda a, b: ridge_fit(a, b, 2.0), ridge_predict, X, y, 5).mean()
    first = got(X, y, [2.0], 5, 4)
    close(first, want, rtol=1e-9, label="nested score with the single penalty 2.0")
    close(got(X, y, [2.0], 5, 4), first, atol=1e-12, label="a second call")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0])
    score = got(X, y, [0.0, 10.0, 1000.0], 5, 4)
    close(score, 0.0, atol=1e-12, label="nested score when penalty 0 fits exactly")


@check("on pure noise the best score of 20 penalties is too good, and the nested score is not")
def test_nested_exceeds_naive(sub):
    got = has(sub, "nested_cv_score")
    cv = has(sub, "cross_val_score")
    lambdas = np.logspace(-2, 3, 20)
    naive, nested = [], []
    for s in range(10):
        rng = np.random.default_rng(100 + s)
        X = rng.normal(size=(60, 10))
        y = rng.normal(size=60)
        naive.append(min(
            cv(lambda a, b, l=l: ridge_fit(a, b, l), ridge_predict, X, y, 5).mean() for l in lambdas
        ))
        nested.append(got(X, y, lambdas, 5, 4))
    if np.mean(nested) < np.mean(naive) + 0.015:
        fail(f"averaged over 10 noise datasets, the lowest of 20 scores was {np.mean(naive):.3f} and the "
             f"nested score was {np.mean(nested):.3f}; the nested score should be clearly higher")
