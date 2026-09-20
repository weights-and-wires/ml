"""Checks for the generalisation and model selection exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def ridge_predict(w, X):
    return X @ w


def regression_dataset(n=60, d=5, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + noise * rng.normal(size=n)
    return X, y, w_true


def ill_conditioned_dataset(n=60, d=30, seed=7):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = Z @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = 0.3 * rng.normal(size=d)
    y = X @ w_true + rng.normal(size=n)
    return X, y, w_true, loadings


@check("k_fold_indices partitions the data exactly once, into nearly equal folds")
def test_k_fold_indices(sub):
    got = has(sub, "k_fold_indices")
    rng = np.random.default_rng(3)
    n, k = 23, 5
    folds = got(n, k, rng)
    if len(folds) != k:
        fail(f"expected {k} folds, got {len(folds)}")
    sizes = [len(f) for f in folds]
    if max(sizes) - min(sizes) > 1:
        fail(f"fold sizes should differ by at most one, got {sizes}")
    concatenated = np.sort(np.concatenate(folds))
    if not np.array_equal(concatenated, np.arange(n)):
        fail("the folds do not partition range(n) exactly once")
    for a in range(k):
        for b in range(a + 1, k):
            overlap = np.intersect1d(folds[a], folds[b])
            if overlap.size:
                fail(f"folds {a} and {b} share indices {overlap.tolist()}")


@check("cross_val_score returns one held-out error per fold, using every point once")
def test_cross_val_score(sub):
    got = has(sub, "cross_val_score")
    X, y, _ = regression_dataset(n=40, d=3)
    scores = got(lambda Xt, yt: ridge_fit(Xt, yt, 1.0), ridge_predict, X, y, k=4)
    shape(scores, (4,), label="cross_val_score output")
    if np.any(scores < 0) or not np.all(np.isfinite(scores)):
        fail(f"held-out mean squared errors must be finite and nonnegative, got {scores}")
    # Calling it again must reproduce the same folds and the same scores.
    scores_again = got(lambda Xt, yt: ridge_fit(Xt, yt, 1.0), ridge_predict, X, y, k=4)
    close(scores, scores_again, atol=1e-12, label="cross_val_score should be deterministic")


@check("loo_linear_smoother matches brute-force leave-one-out refitting")
def test_loo_matches_brute_force(sub):
    got = has(sub, "loo_linear_smoother")
    X, y, _ = regression_dataset(n=50, d=4, seed=1)
    n = X.shape[0]

    for lam in (0.0, 0.5, 5.0):
        errors = np.zeros(n)
        idx_all = np.arange(n)
        for i in range(n):
            mask = idx_all != i
            w = ridge_fit(X[mask], y[mask], lam)
            errors[i] = (y[i] - X[i] @ w) ** 2
        brute_force = float(np.mean(errors))
        fast = got(X, y, lam)
        close(
            fast, brute_force, rtol=1e-6, atol=1e-8,
            label=f"leave-one-out MSE at lam={lam}",
        )


@check("learning_curve reports decreasing validation error as training size grows")
def test_learning_curve(sub):
    got = has(sub, "learning_curve")
    X, y, _ = regression_dataset(n=300, d=5, seed=4, noise=1.0)
    train_sizes = [10, 40, 160]
    train_errors, val_errors = got(
        lambda Xt, yt: ridge_fit(Xt, yt, 1.0), ridge_predict, X, y, train_sizes
    )
    shape(train_errors, (3,), label="training errors")
    shape(val_errors, (3,), label="validation errors")
    if val_errors[-1] > val_errors[0]:
        fail(
            "validation error should fall as the training set grows, but it "
            f"went from {val_errors[0]:.3g} at size {train_sizes[0]} to "
            f"{val_errors[-1]:.3g} at size {train_sizes[-1]}"
        )
    if train_errors[0] > val_errors[0]:
        fail(
            "with only 10 training points a linear model should fit them "
            f"almost exactly, so training error ({train_errors[0]:.3g}) should "
            f"be smaller than validation error ({val_errors[0]:.3g})"
        )


@check("empirical_rademacher is near zero for one hypothesis and grows with class size")
def test_rademacher_grows_with_class_size(sub):
    got = has(sub, "empirical_rademacher")
    n = 40
    rng = np.random.default_rng(2)
    single = rng.choice(np.array([-1.0, 1.0]), size=(1, n))
    small = got(single, np.random.default_rng(10), num_draws=4000)
    if abs(small) > 0.15:
        fail(
            f"a single hypothesis has zero true Rademacher complexity, since "
            f"E[sigma] = 0 independent of the hypothesis; the Monte Carlo "
            f"estimate should be small, but it is {small:.3g}"
        )

    many = rng.choice(np.array([-1.0, 1.0]), size=(300, n))
    large = got(many, np.random.default_rng(11), num_draws=4000)
    if large < 4 * max(abs(small), 1e-3):
        fail(
            f"complexity should grow with the number of hypotheses: got "
            f"{small:.3g} for one hypothesis and {large:.3g} for 300, expected "
            "the second to be several times larger"
        )
    if large > 1.0 + 1e-9:
        fail(f"the empirical Rademacher complexity can never exceed 1, got {large:.3g}")


@check("empirical_rademacher reaches its maximum of 1 for a class that shatters the sample")
def test_rademacher_shattering(sub):
    got = has(sub, "empirical_rademacher")
    n = 7
    patterns = np.array(
        [[1.0 if (i >> b) & 1 else -1.0 for b in range(n)] for i in range(2 ** n)]
    )
    value = got(patterns, np.random.default_rng(12), num_draws=3000)
    close(value, 1.0, atol=0.02, label="Rademacher complexity of a shattering class")


@check("cross-validation selects the ridge penalty that minimises true held-out error")
def test_cv_selects_good_lambda(sub):
    X, y, w_true, loadings = ill_conditioned_dataset()
    lambdas = np.logspace(-2, 3, 20)

    cv_means = np.array(
        [
            has(sub, "cross_val_score")(
                lambda Xt, yt, l=l: ridge_fit(Xt, yt, l), ridge_predict, X, y, k=5
            ).mean()
            for l in lambdas
        ]
    )
    chosen = lambdas[int(np.argmin(cv_means))]

    rng = np.random.default_rng(99)
    n_test, d = 5000, X.shape[1]
    X_test = rng.normal(size=(n_test, 5)) @ loadings + 0.5 * rng.normal(size=(n_test, d))
    y_test = X_test @ w_true + rng.normal(size=n_test)

    def true_risk(lam):
        w = ridge_fit(X, y, lam)
        return float(np.mean((y_test - X_test @ w) ** 2))

    risk_chosen = true_risk(chosen)
    risk_none = true_risk(0.0)
    risk_huge = true_risk(1000.0)

    if risk_chosen > 0.85 * risk_none:
        fail(
            f"the CV-chosen penalty (lambda={chosen:.3g}) should beat no "
            f"regularisation by a clear margin on this ill-conditioned problem: "
            f"true risk {risk_chosen:.3g} against {risk_none:.3g} at lambda=0"
        )
    if risk_chosen > 0.85 * risk_huge:
        fail(
            f"the CV-chosen penalty (lambda={chosen:.3g}) should also beat "
            f"heavy over-regularisation: true risk {risk_chosen:.3g} against "
            f"{risk_huge:.3g} at lambda=1000"
        )
