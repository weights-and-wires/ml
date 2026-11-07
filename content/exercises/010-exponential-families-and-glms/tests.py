"""Checks for the exponential families and GLMs exercise."""

from math import factorial, pi, sqrt

import numpy as np

from mlcheck import check, close, fail, shape

ETAS = [-1.5, -0.3, 0.0, 0.7, 2.0]
FAMILIES = ("gaussian", "bernoulli", "poisson")
X_SHOP = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
Y_SHOP = np.array([1.0, 1.0, 2.0])


def softplus_ref(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def dataset(family, n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    w_true = np.array([0.3, 0.5, -0.4])
    eta = X @ w_true
    if family == "gaussian":
        y = eta + rng.normal(size=n)
    elif family == "bernoulli":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return X, y, w_true


@check("log_partition gives eta^2/2, softplus, and e^eta, stays finite for the Bernoulli at +-800, and rejects unknown families")
def test_log_partition(sub):
    eta = np.array([-1.0, 0.0, 1.0, 2.0])
    close(sub.log_partition("gaussian", eta), 0.5 * eta ** 2, atol=1e-10, label="gaussian A")
    close(sub.log_partition("bernoulli", eta), softplus_ref(eta), atol=1e-10, label="bernoulli A")
    close(sub.log_partition("poisson", eta), np.exp(eta), atol=1e-8, label="poisson A")
    out = sub.log_partition("bernoulli", np.array([-800.0, 800.0]))
    if not np.all(np.isfinite(out)):
        fail(f"bernoulli A at eta = -800 and 800 returned {out}, expected finite values")
    close(out, np.array([0.0, 800.0]), atol=1e-9, label="bernoulli A at large |eta|")
    try:
        sub.log_partition("cauchy", np.array([0.0]))
    except ValueError:
        return
    fail("log_partition should raise ValueError for an unknown family")


@check("log_partition is the log of the total, so the probabilities add up to 1")
def test_normaliser(sub):
    for eta in (-1.2, 0.0, 0.9):
        e = np.array([eta])
        # Bernoulli: h(y) = 1 for y in {0, 1}.
        y = np.array([0.0, 1.0])
        total = np.sum(np.exp(eta * y - sub.log_partition("bernoulli", e)))
        if abs(total - 1.0) > 1e-9:
            fail(f"bernoulli probabilities at eta={eta} add up to {total:.6f}, expected 1")
        # Poisson: h(y) = 1 / y!.
        y = np.arange(0, 80)
        h = np.array([1.0 / factorial(int(k)) if k < 25 else 0.0 for k in y])
        total = np.sum(h * np.exp(eta * y - sub.log_partition("poisson", e)))
        if abs(total - 1.0) > 1e-9:
            fail(f"poisson probabilities at eta={eta} add up to {total:.6f}, expected 1")
        # Gaussian: h(y) = exp(-y^2 / 2) / sqrt(2 pi), integrated on a grid.
        grid = np.linspace(-12.0, 12.0, 4801)
        dens = np.exp(-grid ** 2 / 2) / sqrt(2 * pi) * np.exp(eta * grid - sub.log_partition("gaussian", e))
        total = np.sum(dens) * (grid[1] - grid[0])
        if abs(total - 1.0) > 1e-6:
            fail(f"gaussian density at eta={eta} integrates to {total:.6f}, expected 1")


@check("mean_from_natural is the slope of log_partition and the average y, and stays finite at +-800")
def test_mean(sub):
    eps = 1e-6
    for family in FAMILIES:
        for e0 in ETAS:
            numeric = (sub.log_partition(family, np.array([e0 + eps]))[0]
                       - sub.log_partition(family, np.array([e0 - eps]))[0]) / (2 * eps)
            got = float(sub.mean_from_natural(family, np.array([e0]))[0])
            if abs(got - numeric) > 1e-4 * max(1.0, abs(numeric)):
                fail(f"{family} mean at eta={e0}: got {got:.6g}, the slope of log_partition is {numeric:.6g}")
    shape(sub.mean_from_natural("poisson", np.zeros((2, 3))), (2, 3), label="mean output")
    for eta in (-0.7, 0.0, 1.1):
        e = np.array([eta])
        y = np.arange(0, 25)
        h = np.array([1.0 / factorial(int(k)) for k in y])
        p = h * np.exp(eta * y - sub.log_partition("poisson", e))
        close(sub.mean_from_natural("poisson", e)[0], np.sum(y * p), atol=1e-8, label=f"poisson mean at eta={eta}")
        pb = np.exp(eta * np.array([0.0, 1.0]) - sub.log_partition("bernoulli", e))
        close(sub.mean_from_natural("bernoulli", e)[0], pb[1], atol=1e-10, label=f"bernoulli mean at eta={eta}")
    big = np.array([-800.0, 800.0])
    try:
        with np.errstate(over="raise"):
            m = sub.mean_from_natural("bernoulli", big)
    except FloatingPointError:
        fail("the Bernoulli mean overflowed at eta = -800 or 800; write the sigmoid so that np.exp never sees a large positive number")
    close(m, np.array([0.0, 1.0]), atol=1e-12, label="bernoulli mean at extremes")


@check("variance_from_natural is the bend of log_partition, and the Poisson variance equals its mean")
def test_variance(sub):
    eps = 1e-3
    for family in FAMILIES:
        for e0 in ETAS:
            def A(v):
                return sub.log_partition(family, np.array([v]))[0]
            numeric = (A(e0 + eps) - 2 * A(e0) + A(e0 - eps)) / eps ** 2
            got = float(sub.variance_from_natural(family, np.array([e0]))[0])
            if abs(got - numeric) > 5e-3 * max(1.0, abs(numeric)):
                fail(f"{family} variance at eta={e0}: got {got:.6g}, the bend of log_partition is {numeric:.6g}")
    for eta in (-0.7, 0.0, 1.1):
        e = np.array([eta])
        y = np.arange(0, 25)
        h = np.array([1.0 / factorial(int(k)) for k in y])
        p = h * np.exp(eta * y - sub.log_partition("poisson", e))
        mean = np.sum(y * p)
        close(sub.variance_from_natural("poisson", e)[0], np.sum((y - mean) ** 2 * p), atol=1e-8,
              label=f"poisson variance at eta={eta}")
    eta = np.array([-2.0, -0.5, 0.0, 0.5, 1.5])
    shape(sub.variance_from_natural("gaussian", eta), (5,), label="gaussian variance output (return an array, not the bare number 1)")
    close(sub.variance_from_natural("poisson", eta), sub.mean_from_natural("poisson", eta),
          atol=1e-10, label="Poisson variance should equal Poisson mean")
    close(sub.variance_from_natural("gaussian", eta), np.ones(5), atol=1e-12, label="gaussian variance")
    big = np.array([-800.0, 800.0])
    try:
        with np.errstate(over="raise"):
            v = sub.variance_from_natural("bernoulli", big)
    except FloatingPointError:
        fail("the Bernoulli variance overflowed at eta = -800 or 800")
    close(v, np.array([0.0, 0.0]), atol=1e-12, label="bernoulli variance at extremes")


@check("glm_loss is 3 for the shop counts at zero weights, half the squared error for the Gaussian, and the log loss for the Bernoulli")
def test_loss(sub):
    got = sub.glm_loss("poisson", np.zeros(2), X_SHOP, Y_SHOP)
    close(np.array(got), np.array(3.0), atol=1e-12, label="Poisson loss at zero weights")
    w = np.array([0.05, 0.1])
    eta = X_SHOP @ w
    close(np.array(sub.glm_loss("poisson", w, X_SHOP, Y_SHOP)), np.array(np.sum(np.exp(eta) - Y_SHOP * eta)),
          atol=1e-12, label="Poisson loss at w = (0.05, 0.1)")
    X, y, _ = dataset("gaussian", n=40)
    w = np.array([0.3, -0.2, 0.1])
    want = 0.5 * np.sum((X @ w - y) ** 2) - 0.5 * np.sum(y ** 2)
    close(np.array(sub.glm_loss("gaussian", w, X, y)), np.array(want), atol=1e-9, label="Gaussian loss")
    X, y, _ = dataset("bernoulli", n=40)
    z = X @ w
    close(np.array(sub.glm_loss("bernoulli", w, X, y)), np.array(np.sum(softplus_ref(z) - y * z)),
          atol=1e-9, label="Bernoulli loss")


@check("glm_gradient is [-1, -2] on the shop data and matches finite differences for every family")
def test_gradient(sub):
    g = sub.glm_gradient("poisson", np.zeros(2), X_SHOP, Y_SHOP)
    shape(g, (2,), label="glm_gradient output")
    close(g, np.array([-1.0, -2.0]), atol=1e-12, label="shop gradient")
    for family in FAMILIES:
        X, y, _ = dataset(family, n=40)
        w = np.array([0.1, 0.2, -0.1])
        eps = 1e-6
        numeric = np.zeros(3)
        for j in range(3):
            step = np.zeros(3)
            step[j] = eps
            numeric[j] = (sub.glm_loss(family, w + step, X, y) - sub.glm_loss(family, w - step, X, y)) / (2 * eps)
        got = sub.glm_gradient(family, w, X, y)
        shape(got, (3,), label=f"{family} gradient")
        close(got, numeric, rtol=1e-5, atol=1e-4, label=f"{family} gradient")


@check("glm_fisher_information is [[3, 3], [3, 5]] on the shop data and matches a finite-difference bend of the gradient")
def test_info(sub):
    info = sub.glm_fisher_information("poisson", np.zeros(2), X_SHOP)
    shape(info, (2, 2), label="glm_fisher_information output")
    close(info, np.array([[3.0, 3.0], [3.0, 5.0]]), atol=1e-12, label="shop information")
    close(info, info.T, atol=1e-12, label="information should be symmetric")
    for family in FAMILIES:
        X, y, _ = dataset(family)
        w = np.array([0.1, 0.2, -0.1])
        eps = 1e-5
        numeric = np.zeros((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = eps
            numeric[:, j] = (sub.glm_gradient(family, w + step, X, y)
                             - sub.glm_gradient(family, w - step, X, y)) / (2 * eps)
        got = sub.glm_fisher_information(family, w, X)
        close(got, numeric, rtol=1e-4, atol=1e-4, label=f"{family} information")


@check("newton_step gives [1/6, -0.5] on the shop data and the hand answers for coins and for Gaussian rows")
def test_step(sub):
    delta = sub.newton_step("poisson", np.zeros(2), X_SHOP, Y_SHOP)
    shape(delta, (2,), label="newton_step output")
    close(delta, np.array([1.0 / 6.0, -0.5]), atol=1e-10, label="shop step delta")
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    close(sub.newton_step("bernoulli", np.zeros(2), X, np.array([1.0, 0.0])),
          np.array([-2.0, 2.0]), atol=1e-10, label="Bernoulli step delta")
    close(sub.newton_step("gaussian", np.zeros(2), X, np.array([3.0, 1.0])),
          np.array([-3.0, 1.0]), atol=1e-10, label="Gaussian step delta")


@check("fit_fisher_scoring reproduces the shop iterates and recovers known Poisson weights in under ten iterations")
def test_fit_poisson(sub):
    w, history = sub.fit_fisher_scoring("poisson", X_SHOP, Y_SHOP, max_iter=30)
    close(np.array(history[0]), np.array(3.0), atol=1e-12, label="history[0]")
    close(np.array(history[1]), np.array(2.7097367), atol=1e-6, label="history[1]")
    close(np.array(history[2]), np.array(2.6599146), atol=1e-6, label="history[2]")
    close(w, np.array([-0.14507, 0.38416]), atol=1e-4, label="final shop weights")
    X, y, w_true = dataset("poisson", n=400)
    w, history = sub.fit_fisher_scoring("poisson", X, y, max_iter=30)
    shape(w, (3,), label="fit_fisher_scoring weights")
    if len(history) >= 10:
        fail(f"took {len(history)} iterations, expected fewer than 10")
    close(w, w_true, atol=0.15, label="recovered Poisson weights")
    if any(b > a + 1e-10 for a, b in zip(history, history[1:])):
        fail("the loss should fall at every iteration on this problem")


@check("the Gaussian case takes one step, and the Bernoulli case starts with the logistic Newton step")
def test_gaussian_and_bernoulli(sub):
    X, y, _ = dataset("gaussian")
    w, history = sub.fit_fisher_scoring("gaussian", X, y, max_iter=10)
    if len(history) > 3:
        fail(f"Gaussian fit recorded {len(history)} losses, expected 2 (one step reaches the bottom)")
    close(w, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-8, label="Gaussian weights vs least squares")

    X, y, _ = dataset("bernoulli", n=200)
    w1, _ = sub.fit_fisher_scoring("bernoulli", X, y, max_iter=1)
    g0 = X.T @ (0.5 - y)
    h0 = 0.25 * (X.T @ X)
    close(w1, -np.linalg.solve(h0, g0), atol=1e-8, label="first Bernoulli step")
    w_full, history = sub.fit_fisher_scoring("bernoulli", X, y, max_iter=30)
    if len(history) >= 10:
        fail(f"Bernoulli fit took {len(history)} iterations, expected fewer than 10")
    if np.linalg.norm(sub.glm_gradient("bernoulli", w_full, X, y)) > 1e-4:
        fail("the Bernoulli gradient at the returned weights should be near 0")
