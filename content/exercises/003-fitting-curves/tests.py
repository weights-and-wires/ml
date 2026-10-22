"""Checks for the fitting-curves exercise."""

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


@check("polynomial_features has one column per power, with ones in column 0")
def test_features(sub):
    features = has(sub, "polynomial_features")
    x = np.array([0.0, 1.0, 2.0, -1.0])
    phi = features(x, 3)
    shape(phi, (4, 4), label="polynomial_features output")
    want = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8], [1, -1, 1, -1]], dtype=float)
    close(phi, want, atol=1e-12, label="powers of 0, 1, 2, and -1 (the size 0 row keeps its 1)")


@check("The design matrix times the weights gives the guesses 9 and -1.5")
def test_hand_example(sub):
    features = has(sub, "polynomial_features")
    guesses = features(np.array([2.0, -1.0]), 2) @ np.array([1.0, 3.0, 0.5])
    close(guesses, np.array([9.0, -1.5]), atol=1e-12, label="guesses for sizes 2 and -1")
    predict = has(sub, "predict_polynomial")
    close(predict(np.array([1.0, 3.0, 0.5]), np.array([2.0, -1.0])), np.array([9.0, -1.5]), atol=1e-12,
          label="predict_polynomial on the same example")


@check("fit_polynomial recovers an exact line and an exact quadratic")
def test_fit_exact(sub):
    fit = has(sub, "fit_polynomial")
    w = fit(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]), 1)
    shape(w, (2,), label="fit_polynomial output for degree 1")
    close(w, np.array([1.0, 2.0]), atol=1e-9, label="line through (0,1), (1,3), (2,5)")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=40)
    true_w = np.array([1.0, -2.0, 0.5])
    y = true_w[0] + true_w[1] * x + true_w[2] * x**2
    close(fit(x, y, 2), true_w, atol=1e-8, label="weights of an exact quadratic")


@check("fit_polynomial returns the weights with the smallest total squared miss")
def test_fit_is_minimum(sub):
    fit = has(sub, "fit_polynomial")
    features = has(sub, "polynomial_features")
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


@check("Training risk never rises with degree, and held-out risk falls and then rises")
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


@check("A degree near the number of flats gives tiny training risk and huge held-out risk")
def test_interpolation(sub):
    sweep = has(sub, "sweep_degrees")
    x_train, y_train, x_test, y_test = seeded_flats()
    train_risks, test_risks = sweep(x_train, y_train, x_test, y_test, [12])
    if train_risks[0] > 0.01:
        fail(f"degree 12 on 15 flats should fit them almost exactly, training risk was {train_risks[0]:.4g}")
    if test_risks[0] < 100 * train_risks[0]:
        fail(f"held-out risk ({test_risks[0]:.4g}) should be far above training risk ({train_risks[0]:.4g})")
