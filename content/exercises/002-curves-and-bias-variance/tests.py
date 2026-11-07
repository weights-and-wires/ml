"""Checks for the curves and bias-variance exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def wave(x):
    return np.sin(2.0 * np.pi * x)


def seeded_flats():
    rng = np.random.default_rng(0)
    x_train = rng.uniform(0.0, 1.0, size=15)
    y_train = wave(x_train) + 0.3 * rng.normal(size=15)
    x_test = rng.uniform(0.0, 1.0, size=200)
    y_test = wave(x_test) + 0.3 * rng.normal(size=200)
    return x_train, y_train, x_test, y_test


@check("polynomial_features has one column per power, and the design matrix times the weights gives 9 and -1.5")
def test_features(sub):
    features = has(sub, "polynomial_features")
    x = np.array([0.0, 1.0, 2.0, -1.0])
    phi = features(x, 3)
    shape(phi, (4, 4), label="polynomial_features output")
    want = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8], [1, -1, 1, -1]], dtype=float)
    close(phi, want, atol=1e-12, label="powers of 0, 1, 2, and -1 (the size 0 row keeps its 1)")
    guesses = features(np.array([2.0, -1.0]), 2) @ np.array([1.0, 3.0, 0.5])
    close(guesses, np.array([9.0, -1.5]), atol=1e-12, label="guesses for sizes 2 and -1")
    predict = has(sub, "predict_polynomial")
    close(predict(np.array([1.0, 3.0, 0.5]), np.array([2.0, -1.0])), np.array([9.0, -1.5]), atol=1e-12,
          label="predict_polynomial on the same example")


@check("fit_polynomial recovers exact polynomials and returns the weights with the smallest total squared miss")
def test_fit(sub):
    fit = has(sub, "fit_polynomial")
    features = has(sub, "polynomial_features")
    w = fit(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]), 1)
    shape(w, (2,), label="fit_polynomial output for degree 1")
    close(w, np.array([1.0, 2.0]), atol=1e-9, label="line through (0,1), (1,3), (2,5)")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=40)
    true_w = np.array([1.0, -2.0, 0.5])
    y = true_w[0] + true_w[1] * x + true_w[2] * x**2
    close(fit(x, y, 2), true_w, atol=1e-8, label="weights of an exact quadratic")
    x_train, y_train, _, _ = seeded_flats()
    w = fit(x_train, y_train, 3)
    phi = features(x_train, 3)
    best = np.sum((phi @ w - y_train) ** 2)
    rng = np.random.default_rng(1)
    for _ in range(50):
        nudged = w + 0.05 * rng.normal(size=w.shape)
        if np.sum((phi @ nudged - y_train) ** 2) < best - 1e-12:
            fail("found nearby weights with a lower total squared miss, so these are not least-squares weights")
    # At the minimum, the misses are at right angles to every column.
    close(phi.T @ (phi @ w - y_train), np.zeros(4), atol=1e-8, label="phi.T @ misses")


@check("sweep_degrees matches an independent fit at every degree")
def test_sweep_values(sub):
    sweep = has(sub, "sweep_degrees")
    x_train, y_train, x_test, y_test = seeded_flats()
    degrees = [1, 3, 6]
    train_risks, test_risks = sweep(x_train, y_train, x_test, y_test, degrees)
    shape(train_risks, (3,), label="train_risks")
    shape(test_risks, (3,), label="test_risks")
    for i, d in enumerate(degrees):
        phi_train = np.vander(x_train, d + 1, increasing=True)
        phi_test = np.vander(x_test, d + 1, increasing=True)
        w, *_ = np.linalg.lstsq(phi_train, y_train, rcond=None)
        almost(train_risks[i], np.mean((phi_train @ w - y_train) ** 2), tol=1e-9, label=f"training risk at degree {d}")
        almost(test_risks[i], np.mean((phi_test @ w - y_test) ** 2), tol=1e-9, label=f"held-out risk at degree {d}")


@check("Training risk never rises with degree, held-out risk falls and then explodes near the number of flats")
def test_sweep_shape(sub):
    sweep = has(sub, "sweep_degrees")
    x_train, y_train, x_test, y_test = seeded_flats()
    degrees = list(range(1, 13))
    train_risks, test_risks = sweep(x_train, y_train, x_test, y_test, degrees)
    if np.any(np.diff(train_risks) > 1e-9):
        fail(f"training risk rose with degree: {np.round(train_risks, 4)}")
    best = degrees[int(np.argmin(test_risks))]
    if not 2 <= best <= 6:
        fail(f"the lowest held-out risk should sit at a middle degree (2 to 6), got degree {best}")
    if test_risks[-1] < 10 * test_risks.min():
        fail(f"held-out risk at degree 12 ({test_risks[-1]:.3g}) should be at least 10 times the best ({test_risks.min():.3g})")
    if train_risks[-1] > 0.01:
        fail(f"degree 12 on 15 flats should fit them almost exactly, training risk was {train_risks[-1]:.4g}")
    if test_risks[-1] < 100 * train_risks[-1]:
        fail(f"held-out risk ({test_risks[-1]:.4g}) should be far above training risk ({train_risks[-1]:.4g})")


@check("dart_decomposition gives 0.01, 0.05, and 0.09 for the four hits")
def test_darts(sub):
    darts = has(sub, "dart_decomposition")
    bias_sq, variance, noise = darts(np.array([0.8, 1.2, 1.0, 1.4]), 1.0, 0.3)
    almost(bias_sq, 0.01, tol=1e-12, label="squared bias (mean 1.1 against bullseye 1.0)")
    almost(variance, 0.05, tol=1e-12, label="variance (0.20 divided by 4 hits)")
    almost(noise, 0.09, tol=1e-12, label="noise (0.3 squared)")
    # Two hits at 1 and 3: the variance divides by 2, not by 1.
    _, variance, _ = darts(np.array([1.0, 3.0]), 2.0, 0.5)
    almost(variance, 1.0, tol=1e-12, label="variance of hits 1 and 3 (divide by the number of hits)")


@check("sample_training_set draws sizes in the domain, then noisy labels, from the generator it is given")
def test_sampler(sub):
    sample = has(sub, "sample_training_set")
    x, y = sample(wave, 0.0, 50, (0.0, 1.0), np.random.default_rng(0))
    shape(x, (50,), label="x")
    shape(y, (50,), label="y")
    if x.min() < 0.0 or x.max() > 1.0:
        fail(f"sizes must lie in (0, 1), got range {x.min():.3f} to {x.max():.3f}")
    close(y, wave(x), atol=1e-12, label="labels with noise_std = 0 equal true_function(x)")
    x1, y1 = sample(wave, 0.3, 50, (0.0, 1.0), np.random.default_rng(5))
    x2, y2 = sample(wave, 0.3, 50, (0.0, 1.0), np.random.default_rng(5))
    close(x1, x2, atol=0.0, label="same generator state gives the same sizes")
    close(y1, y2, atol=0.0, label="same generator state gives the same labels")
    rng = np.random.default_rng(3)
    xa, _ = sample(wave, 0.3, 50, (0.0, 1.0), rng)
    xb, _ = sample(wave, 0.3, 50, (0.0, 1.0), rng)
    if np.array_equal(xa, xb):
        fail("two calls with one generator returned the same sizes; draw from the generator you are given")
    x3, y3 = sample(wave, 0.3, 20000, (2.0, 3.0), np.random.default_rng(1))
    resid = y3 - wave(x3)
    almost(resid.mean(), 0.0, tol=0.01, label="mean label noise")
    almost(resid.std(), 0.3, tol=0.01, label="standard deviation of label noise")


@check("bias_variance_decomposition returns three arrays of shape (q,), and a fit with the right shape and no noise has zero bias and variance")
def test_shapes_and_exact(sub):
    decompose = has(sub, "bias_variance_decomposition")
    x_query = np.linspace(0.1, 0.9, 4)
    out = decompose(wave, 0.3, x_query, 2, 20, 30, (0.0, 1.0), 1)
    if len(out) != 3:
        fail(f"expected (bias_sq, variance, noise), got {len(out)} values")
    for name, arr in zip(("bias_sq", "variance", "noise"), out):
        shape(arr, (4,), label=name)
    close(out[2], np.full(4, 0.09), atol=1e-12, label="noise equals noise_std ** 2")
    if np.any(out[0] < 0) or np.any(out[1] < 0):
        fail("squared bias and variance must not be negative")
    quadratic = lambda x: 1.0 - 2.0 * x + 0.5 * x**2
    bias_sq, variance, noise = decompose(quadratic, 0.0, np.array([0.2, 0.5, 0.8]), 2, 20, 40, (0.0, 1.0), 0)
    close(bias_sq, np.zeros(3), atol=1e-8, label="bias_sq of an exact model")
    close(variance, np.zeros(3), atol=1e-8, label="variance of an exact model")
    close(noise, np.zeros(3), atol=1e-12, label="noise with noise_std = 0")


@check("A constant fit to f(x) = x on (0, 1) has bias (0.5 - x) squared and variance (1/12 + noise) / n")
def test_known_answer(sub):
    decompose = has(sub, "bias_variance_decomposition")
    n_train, noise_std = 20, 0.3
    x_query = np.array([0.25, 0.5, 0.9])
    bias_sq, variance, _ = decompose(lambda x: x, noise_std, x_query, 0, n_train, 3000, (0.0, 1.0), 0)
    close(bias_sq, (0.5 - x_query) ** 2, atol=0.004, label="squared bias of the constant fit")
    close(variance, np.full(3, (1.0 / 12.0 + noise_std**2) / n_train), atol=0.0015,
          label="variance of the constant fit")


@check("bias + variance + noise matches the average squared miss against fresh labels")
def test_identity(sub):
    decompose = has(sub, "bias_variance_decomposition")
    fit = has(sub, "fit_polynomial")
    features = has(sub, "polynomial_features")
    x_query = np.linspace(0.1, 0.9, 5)
    degree, n_train, n_trials, noise_std = 3, 20, 1500, 0.25
    bias_sq, variance, noise = decompose(wave, noise_std, x_query, degree, n_train, n_trials, (0.0, 1.0), 0)
    rng = np.random.default_rng(999)
    phi_query = features(x_query, degree)
    misses = np.empty((n_trials, len(x_query)))
    for t in range(n_trials):
        x_train = rng.uniform(0.0, 1.0, size=n_train)
        y_train = wave(x_train) + noise_std * rng.normal(size=n_train)
        w = fit(x_train, y_train, degree)
        fresh = wave(x_query) + noise_std * rng.normal(size=len(x_query))
        misses[t] = (phi_query @ w - fresh) ** 2
    close(bias_sq + variance + noise, misses.mean(axis=0), atol=0.02, label="sum of the three parts against a direct estimate")


@check("Going from degree 1 to degree 7 lowers the bias and raises the variance")
def test_dial(sub):
    decompose = has(sub, "bias_variance_decomposition")
    x_query = np.linspace(0.1, 0.9, 7)
    b1, v1, _ = decompose(wave, 0.2, x_query, 1, 25, 800, (0.0, 1.0), 0)
    b7, v7, _ = decompose(wave, 0.2, x_query, 7, 25, 800, (0.0, 1.0), 0)
    if b7.mean() >= b1.mean():
        fail(f"mean squared bias should fall from degree 1 ({b1.mean():.4g}) to degree 7 ({b7.mean():.4g})")
    if v7.mean() <= v1.mean():
        fail(f"mean variance should rise from degree 1 ({v1.mean():.4g}) to degree 7 ({v7.mean():.4g})")
