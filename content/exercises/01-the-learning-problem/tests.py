"""Checks for the learning-problem exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def sine(x: np.ndarray) -> np.ndarray:
    """The regression function used across several checks."""
    return np.sin(2.0 * np.pi * x)


@check("empirical_risk averages a per-sample loss")
def test_empirical_risk_basic(sub):
    got = has(sub, "empirical_risk")
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.5, 2.5, 2.0, 5.0])
    squared_error = lambda p, t: (p - t) ** 2
    want = float(np.mean((predictions - targets) ** 2))
    almost(got(predictions, targets, squared_error), want, tol=1e-10, label="mean squared error")

    zero_one = lambda p, t: (p != t).astype(float)
    almost(
        got(np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]), zero_one),
        2.0 / 3.0,
        tol=1e-10,
        label="mean zero-one loss",
    )


@check("polynomial_features builds [1, x, x^2, ..., x^degree]")
def test_polynomial_features(sub):
    got = has(sub, "polynomial_features")
    x = np.array([0.0, 1.0, 2.0, -1.0])
    phi = got(x, 3)
    shape(phi, (4, 4), label="polynomial_features output")
    want = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0, 8.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    close(phi, want, atol=1e-10, label="Vandermonde columns")


@check("fit_polynomial recovers an exact polynomial with no noise")
def test_fit_polynomial(sub):
    got = has(sub, "fit_polynomial")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=50)
    true_w = np.array([1.0, -2.0, 0.5])
    y = true_w[0] + true_w[1] * x + true_w[2] * x**2
    w = got(x, y, 2)
    shape(w, (3,), label="fit_polynomial output")
    close(w, true_w, atol=1e-6, label="recovered coefficients")


@check("train_test_split partitions every index exactly once, reproducibly")
def test_train_test_split(sub):
    got = has(sub, "train_test_split")
    x = np.arange(20, dtype=float)
    y = x * 2.0

    x_train, y_train, x_test, y_test = got(x, y, test_size=0.25, seed=0)
    if x_train.shape[0] != 15 or x_test.shape[0] != 5:
        fail(
            f"expected 15 training and 5 held-out points from test_size=0.25 on "
            f"n=20, got {x_train.shape[0]} and {x_test.shape[0]}"
        )
    close(y_train, x_train * 2.0, atol=0.0, label="labels track inputs after the split")
    close(y_test, x_test * 2.0, atol=0.0, label="labels track inputs after the split")

    combined = np.sort(np.concatenate([x_train, x_test]))
    close(combined, x, atol=0.0, label="union of the two splits recovers every index")

    x_train_again, _, x_test_again, _ = got(x, y, test_size=0.25, seed=0)
    close(x_train_again, x_train, atol=0.0, label="same seed reproduces the same split")
    close(x_test_again, x_test, atol=0.0, label="same seed reproduces the same split")


@check("The training risk underestimates the risk on a held-out split")
def test_train_risk_underestimates_test_risk(sub):
    split = has(sub, "train_test_split")
    fit = has(sub, "fit_polynomial")
    features = has(sub, "polynomial_features")
    risk = has(sub, "empirical_risk")

    rng = np.random.default_rng(0)
    n, degree = 60, 25
    x = rng.uniform(-1.0, 1.0, size=n)
    y = sine(x) + 0.3 * rng.normal(size=n)

    x_train, y_train, x_test, y_test = split(x, y, test_size=0.4, seed=0)
    w = fit(x_train, y_train, degree)
    squared_error = lambda p, t: (p - t) ** 2

    train_risk = risk(features(x_train, degree) @ w, y_train, squared_error)
    test_risk = risk(features(x_test, degree) @ w, y_test, squared_error)

    if train_risk >= test_risk:
        fail(
            "fitting a high-degree polynomial to a small training set should make "
            f"the training risk ({train_risk:.4g}) far smaller than the risk on "
            f"held-out data ({test_risk:.4g}), but it was not"
        )


@check("The bias-variance decomposition sums to the total expected squared error")
def test_bias_variance_sums_to_total(sub):
    decompose = has(sub, "bias_variance_decomposition")
    fit = has(sub, "fit_polynomial")
    features = has(sub, "polynomial_features")

    x_query = np.linspace(-0.6, 0.6, 5)
    degree, n_train, n_trials, noise_std = 3, 20, 1500, 0.25

    bias_sq, variance, noise = decompose(
        sine, noise_std, x_query, degree, n_train, n_trials, seed=0
    )
    shape(bias_sq, (5,), label="bias_sq")
    shape(variance, (5,), label="variance")
    shape(noise, (5,), label="noise")
    predicted_total = bias_sq + variance + noise

    # An independent Monte Carlo estimate of E[(y - fhat(x))^2], built without
    # calling bias_variance_decomposition, to check the identity rather than
    # the implementation.
    rng = np.random.default_rng(999)
    phi_query = features(x_query, degree)
    squared_errors = np.empty((n_trials, x_query.shape[0]))
    for trial in range(n_trials):
        x_train = rng.uniform(-1.0, 1.0, size=n_train)
        y_train = sine(x_train) + noise_std * rng.normal(size=n_train)
        w = fit(x_train, y_train, degree)
        prediction = phi_query @ w
        fresh_label = sine(x_query) + noise_std * rng.normal(size=x_query.shape[0])
        squared_errors[trial] = (prediction - fresh_label) ** 2
    direct_total = squared_errors.mean(axis=0)

    close(
        predicted_total,
        direct_total,
        atol=0.08,
        label="bias^2 + variance + noise against a direct Monte Carlo estimate",
    )


@check("The noise term is the label noise variance, independent of the fit")
def test_noise_floor(sub):
    decompose = has(sub, "bias_variance_decomposition")
    x_query = np.linspace(-0.5, 0.5, 4)
    _, _, noise = decompose(sine, 0.3, x_query, degree=2, n_train=20, n_trials=50, seed=1)
    close(noise, np.full(4, 0.09), atol=1e-10, label="noise floor equals noise_std ** 2")


@check("Bias falls and variance rises as the polynomial degree grows")
def test_bias_falls_variance_rises(sub):
    decompose = has(sub, "bias_variance_decomposition")
    x_query = np.linspace(-0.7, 0.7, 7)

    bias_low, variance_low, _ = decompose(
        sine, 0.2, x_query, degree=1, n_train=25, n_trials=800, seed=0
    )
    bias_high, variance_high, _ = decompose(
        sine, 0.2, x_query, degree=7, n_train=25, n_trials=800, seed=0
    )

    if bias_high.mean() >= bias_low.mean():
        fail(
            "a degree-7 fit should have far lower bias than a degree-1 fit on a "
            f"sine target, but mean bias^2 went from {bias_low.mean():.4g} "
            f"(degree 1) to {bias_high.mean():.4g} (degree 7)"
        )
    if variance_high.mean() <= variance_low.mean():
        fail(
            "a degree-7 fit on only 25 points should have higher variance than a "
            f"degree-1 fit, but mean variance went from {variance_low.mean():.4g} "
            f"(degree 1) to {variance_high.mean():.4g} (degree 7)"
        )
