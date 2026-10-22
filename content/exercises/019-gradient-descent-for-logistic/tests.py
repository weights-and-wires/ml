"""Checks for the gradient descent exercise."""

import numpy as np

from mlcheck import check, close, fail, finite, has, shape


def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))


def separable_dataset(n: int = 150, d: int = 3, seed: int = 5):
    """A design matrix and labels that a linear boundary separates exactly."""
    rng = np.random.default_rng(seed)
    w_sep = np.array([2.0, -1.5, 1.0])[:d]
    X = rng.normal(size=(n, d))
    margin = X @ w_sep
    while np.abs(margin).min() < 0.05:
        X = rng.normal(size=(n, d))
        margin = X @ w_sep
    y = (margin > 0).astype(float)
    return X, y


def logistic_dataset(n: int = 300, d: int = 4, seed: int = 0):
    """A well-conditioned, non-separable logistic regression problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5, 0.8])[:d]
    y = (rng.uniform(size=n) < sigmoid_ref(X @ w_true)).astype(float)
    return X, y


@check("gradient matches a finite-difference derivative of log_loss")
def test_gradient(sub):
    from mlcheck import finite_difference

    log_loss = has(sub, "log_loss")
    grad = has(sub, "gradient")
    X, y = logistic_dataset(n=60, d=4)
    w = np.array([0.3, -0.2, 0.1, 0.5])
    for ridge in (0.0, 0.5):
        numeric = finite_difference(lambda ww: log_loss(ww, X, y, ridge), w)
        analytic = grad(w, X, y, ridge)
        shape(analytic, (4,), label=f"gradient shape, ridge={ridge}")
        close(analytic, numeric, rtol=1e-5, atol=1e-5, label=f"gradient, ridge={ridge}")


@check("gradient on the two lesson emails is [-0.1951, 0.3649]")
def test_gradient_emails(sub):
    grad = has(sub, "gradient")
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    y = np.array([1.0, 0.0])
    w = np.array([0.5, -1.0])
    close(grad(w, X, y), np.array([-0.19511514, 0.36485105]), atol=1e-7, label="gradient on the emails")
    # A ridge fee adds ridge * w.
    close(grad(w, X, y, 2.0) - grad(w, X, y, 0.0), 2.0 * w, atol=1e-10, label="ridge term of the gradient")


@check("a small step against the gradient lowers the loss")
def test_gradient_direction(sub):
    log_loss = has(sub, "log_loss")
    grad = has(sub, "gradient")
    X, y = logistic_dataset(n=100, d=4, seed=1)
    rng = np.random.default_rng(2)
    for _ in range(5):
        w = rng.normal(size=4)
        g = grad(w, X, y)
        before = log_loss(w, X, y)
        after = log_loss(w - 1e-4 * g, X, y)
        if not after < before:
            fail(f"a step of 1e-4 against the gradient raised the loss from {before:.6f} to {after:.6f}; check the sign")


@check("fit_gradient_descent records the loss before each step, starting at n log 2")
def test_history(sub):
    fit = has(sub, "fit_gradient_descent")
    X, y = logistic_dataset(n=80, d=4, seed=3)
    w, history = fit(X, y, lr=0.002, max_iter=50, tol=0.0)
    shape(w, (4,), label="fitted weights")
    if len(history) != 50:
        fail(f"expected 50 history entries for max_iter=50 and tol=0, got {len(history)}")
    close(history[0], 80 * np.log(2.0), atol=1e-8, label="history[0], the loss at zero weights")
    if any(b > a + 1e-9 for a, b in zip(history, history[1:])):
        fail("with a small learning rate the loss should never rise")


@check("fit_gradient_descent stops when the gradient is shorter than tol")
def test_tolerance(sub):
    fit = has(sub, "fit_gradient_descent")
    grad = has(sub, "gradient")
    X, y = logistic_dataset(n=80, d=4, seed=3)
    w, history = fit(X, y, lr=0.002, max_iter=100, tol=1e9)
    if len(history) != 1:
        fail(f"a huge tol should stop at once with one history entry, got {len(history)}")
    close(w, np.zeros(4), atol=0.0, label="weights when stopping before the first step")
    w, history = fit(X, y, lr=0.004, max_iter=20000, tol=1e-6)
    if len(history) >= 20000:
        fail("the loop never reached the tolerance in 20000 steps; check the step and the stopping rule")
    norm = float(np.linalg.norm(grad(w, X, y)))
    if norm > 1e-6:
        fail(f"gradient norm at the reported weights is {norm:.3g}, expected below tol 1e-6")


@check("train_accuracy counts correct labels, and a fit beats the majority baseline")
def test_accuracy(sub):
    acc = has(sub, "train_accuracy")
    fit = has(sub, "fit_gradient_descent")
    X2 = np.array([[1.0, 0.0], [1.0, 2.0]])
    y2 = np.array([1.0, 0.0])
    close(acc(np.array([0.5, -1.0]), X2, y2), 1.0, atol=0.0, label="accuracy on the two emails")
    close(acc(np.array([-0.5, 1.0]), X2, y2), 0.0, atol=0.0, label="accuracy with the signs flipped")
    X, y = logistic_dataset(n=300, d=4)
    w, _ = fit(X, y, lr=0.004, max_iter=3000, tol=1e-6)
    got = acc(w, X, y)
    baseline = max(y.mean(), 1.0 - y.mean())
    if not got > baseline + 0.05:
        fail(f"accuracy {got:.3f} should beat the majority baseline {baseline:.3f} by at least 0.05")


@check("on separable data the unpenalised weights keep growing and the ridge weights stay small")
def test_separability(sub):
    X, y = separable_dataset()
    fit = has(sub, "fit_gradient_descent")
    w_early, _ = fit(X, y, lr=0.05, max_iter=300, ridge=0.0, tol=0.0)
    w_late, history_plain = fit(X, y, lr=0.05, max_iter=3000, ridge=0.0, tol=0.0)
    w_ridge, history_ridge = fit(X, y, lr=0.05, max_iter=3000, ridge=1.0, tol=0.0)
    norm_early = float(np.linalg.norm(w_early))
    norm_late = float(np.linalg.norm(w_late))
    norm_ridge = float(np.linalg.norm(w_ridge))
    if norm_late < 1.3 * norm_early:
        fail(
            "the unpenalised weight norm should keep growing on separable data: "
            f"{norm_early:.3g} at 300 steps but only {norm_late:.3g} at 3000"
        )
    if norm_ridge > 0.5 * norm_late:
        fail(f"the ridge norm ({norm_ridge:.3g}) should stay far below the unpenalised norm ({norm_late:.3g})")
    if history_ridge[-1] < history_plain[-1]:
        fail("the ridge loss should settle above the unpenalised loss, because the fee stops the loss reaching 0")
