"""Checks for the winner's-curse and flexibility exercise."""

import numpy as np

from mlcheck import check, close, fail, has


def ridge_fit(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def ridge_predict(w, X):
    return X @ w


@check("best_of_m_gap is 0 for one model and about 0.187 for the best of 20 with noise 0.1")
def test_gap_values(sub):
    got = has(sub, "best_of_m_gap")
    one = got(1.0, 0.1, 1, 20000, np.random.default_rng(0))
    close(one, 0.0, atol=0.005, label="best of 1 model")
    twenty = got(1.0, 0.1, 20, 20000, np.random.default_rng(0))
    close(twenty, 0.187, atol=0.01, label="best of 20 models (1.87 noise standard deviations)")


@check("best_of_m_gap grows with m, scales with the noise, and ignores the true error")
def test_gap_scaling(sub):
    got = has(sub, "best_of_m_gap")
    gaps = [got(1.0, 0.1, m, 20000, np.random.default_rng(1)) for m in (2, 5, 20, 100)]
    if not all(a < b for a, b in zip(gaps, gaps[1:])):
        fail(f"the gap should grow with m for m = 2, 5, 20, 100, got {np.round(gaps, 3).tolist()}")
    small = got(1.0, 0.1, 20, 20000, np.random.default_rng(2))
    large = got(1.0, 0.2, 20, 20000, np.random.default_rng(2))
    close(large / small, 2.0, rtol=1e-6, label="doubling the noise should double the gap")
    shifted = got(5.0, 0.1, 20, 20000, np.random.default_rng(2))
    close(shifted, small, atol=1e-9, label="the gap should not depend on the true error")


@check("empirical_rademacher is near 0 for one model, grows for 300 models, and never exceeds 1")
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


@check("empirical_rademacher gives 0.5 for the two-model hand example and 1 when every sign pattern is present")
def test_rademacher_exact(sub):
    got = has(sub, "empirical_rademacher")
    two = np.array([[1.0, 1.0], [1.0, -1.0]])
    close(got(two, np.random.default_rng(3), 20000), 0.5, atol=0.01, label="models [1, 1] and [1, -1]")
    n = 7
    patterns = np.array([[1.0 if (i >> b) & 1 else -1.0 for b in range(n)] for i in range(2 ** n)])
    close(got(patterns, np.random.default_rng(12), 3000), 1.0, atol=1e-9,
          label="all 128 sign patterns on 7 examples")


@check("nested_cv_score with one candidate equals the plain cross-validation score of that candidate")
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


@check("nested_cv_score picks the penalty that fits noise-free data exactly")
def test_nested_picks_best(sub):
    got = has(sub, "nested_cv_score")
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
