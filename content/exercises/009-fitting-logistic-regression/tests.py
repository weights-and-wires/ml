"""Checks for the logistic regression fitting exercise."""

import numpy as np

from mlcheck import check, close, fail, finite, finite_difference, has, shape


def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))


EMAILS_X = np.array([[1.0, 0.0], [1.0, 2.0]])
EMAILS_Y = np.array([1.0, 0.0])


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


def numerical_hessian(grad_func, w, eps=1e-5):
    """Central-difference Jacobian of a vector-valued gradient function."""
    d = w.shape[0]
    H = np.zeros((d, d))
    for j in range(d):
        step = np.zeros(d)
        step[j] = eps
        H[:, j] = (grad_func(w + step) - grad_func(w - step)) / (2 * eps)
    return H


@check("gradient matches finite differences of log_loss and the hand result on the two emails")
def test_gradient(sub):
    log_loss = has(sub, "log_loss")
    grad = has(sub, "gradient")
    X, y = logistic_dataset(n=60, d=4)
    w = np.array([0.3, -0.2, 0.1, 0.5])
    for ridge in (0.0, 0.5):
        numeric = finite_difference(lambda ww: log_loss(ww, X, y, ridge), w)
        analytic = grad(w, X, y, ridge)
        shape(analytic, (4,), label=f"gradient shape, ridge={ridge}")
        close(analytic, numeric, rtol=1e-5, atol=1e-5, label=f"gradient, ridge={ridge}")
    w2 = np.array([0.5, -1.0])
    close(grad(w2, EMAILS_X, EMAILS_Y), np.array([-0.19511514, 0.36485105]), atol=1e-7, label="gradient on the emails")
    close(grad(w2, EMAILS_X, EMAILS_Y, 2.0) - grad(w2, EMAILS_X, EMAILS_Y, 0.0), 2.0 * w2, atol=1e-10, label="ridge term of the gradient")
    rng = np.random.default_rng(2)
    X, y = logistic_dataset(n=100, d=4, seed=1)
    for _ in range(5):
        w = rng.normal(size=4)
        before = log_loss(w, X, y)
        after = log_loss(w - 1e-4 * grad(w, X, y), X, y)
        if not after < before:
            fail(f"a step of 1e-4 against the gradient raised the loss from {before:.6f} to {after:.6f}; check the sign")


@check("hessian matches a finite-difference Jacobian of gradient and the hand values on the two emails")
def test_hessian_values(sub):
    grad = has(sub, "gradient")
    hess = has(sub, "hessian")
    X, y = logistic_dataset(n=60, d=4)
    w = np.array([0.3, -0.2, 0.1, 0.5])
    for ridge in (0.0, 0.5):
        numeric = numerical_hessian(lambda ww: grad(ww, X, y, ridge), w)
        analytic = hess(w, X, y, ridge)
        shape(analytic, (4, 4), label=f"hessian shape, ridge={ridge}")
        close(analytic, numeric, rtol=1e-4, atol=1e-4, label=f"hessian, ridge={ridge}")
    got = hess(np.array([0.5, -1.0]), EMAILS_X, EMAILS_Y)
    close(got, np.array([[0.38415016, 0.2982929], [0.2982929, 0.59658581]]), atol=1e-7, label="hessian at (0.5, -1)")
    close(hess(np.zeros(2), EMAILS_X, EMAILS_Y), np.array([[0.5, 0.5], [0.5, 1.0]]), atol=1e-12, label="hessian at zero weights")
    gap = hess(np.zeros(2), EMAILS_X, EMAILS_Y, ridge=0.5) - hess(np.zeros(2), EMAILS_X, EMAILS_Y)
    close(gap, 0.5 * np.eye(2), atol=1e-12, label="ridge adds ridge * I")


@check("hessian is symmetric and positive semidefinite, and confident rows bend the loss less")
def test_hessian_shape_of_bend(sub):
    hess = has(sub, "hessian")
    X, y = logistic_dataset(n=80, d=4, seed=2)
    w = np.random.default_rng(3).normal(size=4)
    H = hess(w, X, y)
    close(H, H.T, atol=1e-10, label="hessian should be symmetric")
    smallest = float(np.linalg.eigvalsh(H).min())
    if smallest < -1e-6:
        fail(f"hessian has a negative eigenvalue {smallest:.3g}, so it is not positive semidefinite")
    traces = [float(np.trace(hess(t * np.array([1.0, -1.0]), EMAILS_X, EMAILS_Y))) for t in (0.0, 1.0, 5.0)]
    if not (traces[0] > traces[1] > traces[2]):
        fail(f"the trace should fall as the emails become more confident (t = 0, 1, 5), got {traces}")


@check("fit_gradient_descent records the loss before each step, starts at n log 2, and stops on tol")
def test_gradient_descent_loop(sub):
    fit = has(sub, "fit_gradient_descent")
    grad = has(sub, "gradient")
    X, y = logistic_dataset(n=80, d=4, seed=3)
    w, history = fit(X, y, lr=0.002, max_iter=50, tol=0.0)
    shape(w, (4,), label="fitted weights")
    if len(history) != 50:
        fail(f"expected 50 history entries for max_iter=50 and tol=0, got {len(history)}")
    close(history[0], 80 * np.log(2.0), atol=1e-8, label="history[0], the loss at zero weights")
    if any(b > a + 1e-9 for a, b in zip(history, history[1:])):
        fail("with a small learning rate the loss should never rise")
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


@check("newton_step solves H delta = g and gives (-2, 2) from zero weights")
def test_newton_step(sub):
    step = has(sub, "newton_step")
    hess = has(sub, "hessian")
    grad = has(sub, "gradient")
    close(step(np.zeros(2), EMAILS_X, EMAILS_Y), np.array([-2.0, 2.0]), atol=1e-10, label="step from zero weights")
    X, y = logistic_dataset(n=60, d=4)
    w = np.array([0.3, -0.2, 0.1, 0.5])
    delta = step(w, X, y, 0.5)
    shape(delta, (4,), label="delta")
    close(hess(w, X, y, 0.5) @ delta, grad(w, X, y, 0.5), rtol=1e-8, atol=1e-8, label="H @ delta versus the gradient")


@check("fit_newton finishes in under 10 passes at a stationary point")
def test_newton_converges(sub):
    fit = has(sub, "fit_newton")
    grad = has(sub, "gradient")
    X, y = logistic_dataset(n=200, d=3, seed=9)
    w, history = fit(X, y, max_iter=30, ridge=1e-6)
    close(history[0], 200 * np.log(2.0), atol=1e-8, label="history[0], the loss at zero weights")
    if len(history) >= 10:
        fail(f"fit_newton took {len(history)} passes, expected fewer than 10")
    residual = float(np.linalg.norm(grad(w, X, y, 1e-6)))
    if residual > 1e-4:
        fail(f"gradient norm at the reported weights is {residual:.3g}, expected near 0")
    if history[-1] >= history[0]:
        fail("the final loss should be below the loss at zero weights")


@check("Newton and gradient descent reach the same weights, Newton in far fewer passes")
def test_newton_vs_descent(sub):
    X, y = logistic_dataset()
    w_newton, h_newton = has(sub, "fit_newton")(X, y, max_iter=30)
    w_gd, h_gd = has(sub, "fit_gradient_descent")(X, y, lr=0.005, max_iter=5000, tol=1e-9)
    if len(h_newton) >= 10:
        fail(f"fit_newton took {len(h_newton)} passes, expected fewer than 10")
    if len(h_gd) < 10 * len(h_newton):
        fail(f"gradient descent took {len(h_gd)} passes, expected at least ten times Newton's {len(h_newton)}")
    close(w_newton, w_gd, atol=1e-3, label="both routes should reach the same weights")


@check("train_accuracy counts correct labels, and a fit beats the majority baseline")
def test_accuracy(sub):
    acc = has(sub, "train_accuracy")
    fit = has(sub, "fit_gradient_descent")
    close(acc(np.array([0.5, -1.0]), EMAILS_X, EMAILS_Y), 1.0, atol=0.0, label="accuracy on the two emails")
    close(acc(np.array([-0.5, 1.0]), EMAILS_X, EMAILS_Y), 0.0, atol=0.0, label="accuracy with the signs flipped")
    X, y = logistic_dataset(n=300, d=4)
    w, _ = fit(X, y, lr=0.004, max_iter=3000, tol=1e-6)
    got = acc(w, X, y)
    baseline = max(y.mean(), 1.0 - y.mean())
    if not got > baseline + 0.05:
        fail(f"accuracy {got:.3f} should beat the majority baseline {baseline:.3f} by at least 0.05")


@check("on separable data the unpenalised weights keep growing and the ridge weights stay small")
def test_separability_descent(sub):
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


@check("on a perfect split Newton marches off without a ridge fee and stops with one, and duplicate columns still solve")
def test_separable_and_duplicates(sub):
    fit = has(sub, "fit_newton")
    w_free, h_free = fit(EMAILS_X, EMAILS_Y)
    finite(w_free, label="weights with the default ridge")
    if np.linalg.norm(w_free) < 10.0:
        fail(f"with no real ridge fee the weights should keep growing on a perfect split, got length {np.linalg.norm(w_free):.3g}")
    w_ridge, h_ridge = fit(EMAILS_X, EMAILS_Y, ridge=1.0)
    close(w_ridge, np.array([0.17604539, -0.56011317]), atol=1e-5, label="weights with ridge 1")
    if len(h_ridge) >= 10:
        fail(f"with ridge 1 Newton should settle in under 10 passes, took {len(h_ridge)}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    X = np.column_stack([X, X[:, 0]])
    y = (rng.uniform(size=100) < sigmoid_ref(X[:, 0] - X[:, 1])).astype(float)
    try:
        w, history = fit(X, y)
    except np.linalg.LinAlgError:
        fail("fit_newton raised a singular-matrix error on duplicate columns; keep the default ridge of 1e-8")
    finite(w, label="weights with duplicate columns")
    close(w[0], w[2], atol=1e-6, label="copies of one column should get the same weight")
