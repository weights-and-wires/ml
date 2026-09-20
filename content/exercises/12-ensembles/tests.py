"""Checks for the bagging and boosting exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


@check("bagging_variance matches its two closed-form extremes")
def test_bagging_variance_extremes(sub):
    got = has(sub, "bagging_variance")
    # rho = 0: independent estimators, variance falls as sigma2 / B.
    almost(got(0.0, 2.0, 10), 2.0 / 10, tol=1e-10, label="rho=0")
    # rho = 1: perfectly correlated estimators, averaging does nothing.
    almost(got(1.0, 2.0, 10), 2.0, tol=1e-10, label="rho=1")
    # B = 1: a single estimator, whatever rho is.
    almost(got(0.4, 3.0, 1), 3.0, tol=1e-10, label="B=1")


@check("bootstrap_indices draws n indices in range with replacement")
def test_bootstrap_indices_shape(sub):
    got = has(sub, "bootstrap_indices")
    rng = np.random.default_rng(0)
    idx = got(50, rng)
    shape(idx, (50,), label="bootstrap_indices output")
    if idx.min() < 0 or idx.max() > 49:
        fail(f"indices must lie in [0, 49], got range [{idx.min()}, {idx.max()}]")
    # With replacement from 50 slots, seeing a repeat is overwhelmingly likely.
    if len(np.unique(idx)) == 50:
        fail("50 draws with replacement from 50 slots should almost never be all-distinct")


@check("the in-bag fraction of bootstrap_indices converges to 1 - 1/e")
def test_in_bag_fraction(sub):
    got = has(sub, "bootstrap_indices")
    rng = np.random.default_rng(1)
    n = 2000
    trials = 60
    fractions = []
    for _ in range(trials):
        idx = got(n, rng)
        fractions.append(len(np.unique(idx)) / n)
    empirical = float(np.mean(fractions))
    want = 1.0 - np.exp(-1.0)
    almost(
        empirical,
        want,
        tol=0.01,
        label=f"in-bag fraction averaged over {trials} trials at n={n}",
    )


@check("bagging_variance matches a Monte Carlo simulation of correlated estimators")
def test_bagging_variance_monte_carlo(sub):
    got = has(sub, "bagging_variance")
    rng = np.random.default_rng(2)
    sigma2 = 1.5
    for rho, B in [(0.2, 5), (0.5, 8), (0.05, 20)]:
        cov = sigma2 * (rho * np.ones((B, B)) + (1 - rho) * np.eye(B))
        draws = rng.multivariate_normal(mean=np.zeros(B), cov=cov, size=20000)
        empirical = float(np.var(draws.mean(axis=1)))
        predicted = got(rho, sigma2, B)
        almost(
            predicted,
            empirical,
            tol=0.03 * sigma2,
            label=f"rho={rho}, B={B}: predicted vs Monte Carlo variance of the average",
        )


@check("fit_bagged_stumps returns one stump and one index array per estimator")
def test_fit_bagged_stumps_shape(sub):
    fit = has(sub, "fit_bagged_stumps")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=100)

    stumps, in_bag = fit(X, y, n_estimators=15, rng=rng)
    if len(stumps) != 15 or len(in_bag) != 15:
        fail(f"expected 15 stumps and 15 index arrays, got {len(stumps)} and {len(in_bag)}")
    for idx in in_bag:
        shape(np.asarray(idx), (100,), label="one bootstrap index array")
    for stump in stumps:
        for key in ("feature", "threshold", "left_value", "right_value"):
            if key not in stump:
                fail(f"a stump returned by fit_bagged_stumps is missing key {key!r}")


@check("out_of_bag_error tracks a genuinely held-out test error")
def test_out_of_bag_error(sub):
    fit = has(sub, "fit_bagged_stumps")
    oob_fn = has(sub, "out_of_bag_error")
    predict_stump = has(sub, "_predict_stump")

    rng = np.random.default_rng(4)
    n = 500
    X = rng.normal(size=(n, 3))
    true_f = lambda X: np.maximum(X[:, 0], 0.0) - 0.5 * X[:, 1]
    y = true_f(X) + 0.3 * rng.normal(size=n)

    X_test = rng.normal(size=(n, 3))
    y_test = true_f(X_test) + 0.3 * rng.normal(size=n)

    stumps, in_bag = fit(X, y, n_estimators=80, rng=rng)
    oob = oob_fn(X, y, stumps, in_bag)

    ensemble_pred = np.mean([predict_stump(s, X_test) for s in stumps], axis=0)
    test_mse = float(np.mean((ensemble_pred - y_test) ** 2))

    if oob <= 0:
        fail(f"out-of-bag error must be positive on noisy data, got {oob}")
    if abs(oob - test_mse) > 0.5 * test_mse + 0.05:
        fail(
            f"out-of-bag error {oob:.4f} should be within a reasonable band of the "
            f"independent test error {test_mse:.4f}, since both estimate the same "
            "generalisation error"
        )


@check("gradient_boost targets the raw labels in its first round, from a zero start")
def test_gradient_boost_first_round(sub):
    boost = has(sub, "gradient_boost")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=80)

    _, history = boost(X, y, loss_grad=lambda yy, F: yy - F, n_rounds=1, learning_rate=0.5)
    if len(history) != 1:
        fail(f"expected a history of length 1 for n_rounds=1, got {len(history)}")
    close(
        history[0]["target"],
        y,
        atol=1e-10,
        label="round-1 target under squared error, starting from F=0",
    )


@check("gradient_boost's training loss decreases monotonically under squared error")
def test_gradient_boost_monotone(sub):
    boost = has(sub, "gradient_boost")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(150, 4))
    y = np.sin(X[:, 0]) + X[:, 1] - 0.5 * X[:, 2] + 0.2 * rng.normal(size=150)

    _, history = boost(X, y, loss_grad=lambda yy, F: yy - F, n_rounds=25, learning_rate=0.4)
    losses = [0.5 * float(np.mean((y - h["prediction"]) ** 2)) for h in history]

    for m in range(len(losses) - 1):
        if losses[m + 1] > losses[m] + 1e-9:
            fail(
                f"training loss increased from {losses[m]:.6f} at round {m + 1} to "
                f"{losses[m + 1]:.6f} at round {m + 2}: {losses}"
            )
    if losses[-1] >= losses[0]:
        fail(
            f"loss should fall substantially over 25 rounds, went from {losses[0]:.6f} "
            f"to {losses[-1]:.6f}"
        )


@check("a boosting round's loss reduction matches eta * (1 - eta/2) * mean(h^2) exactly")
def test_gradient_boost_shrinkage(sub):
    boost = has(sub, "gradient_boost")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 3))
    y = 1.5 * X[:, 0] - X[:, 2] + 0.15 * rng.normal(size=120)
    loss0 = 0.5 * float(np.mean(y**2))  # loss at F = 0

    for eta in (0.1, 0.3, 0.6, 1.0):
        _, history = boost(X, y, loss_grad=lambda yy, F: yy - F, n_rounds=1, learning_rate=eta)
        prediction = history[0]["prediction"]  # = eta * h(X), since F started at 0
        h = prediction / eta
        mean_h2 = float(np.mean(h**2))
        predicted_loss = loss0 - eta * (1.0 - eta / 2.0) * mean_h2
        actual_loss = 0.5 * float(np.mean((y - prediction) ** 2))
        almost(
            actual_loss,
            predicted_loss,
            tol=1e-8 * max(1.0, abs(predicted_loss)),
            label=f"loss after one round at learning_rate={eta}",
        )
