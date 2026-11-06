"""Checks for the Fisher scoring exercise."""

import numpy as np

from mlcheck import check, close, fail, shape

X_SHOP = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
Y_SHOP = np.array([1.0, 1.0, 2.0])


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


@check("glm_fisher_information is [[3, 3], [3, 5]] for the shop data at zero weights")
def test_info_hand(sub):
    info = sub.glm_fisher_information("poisson", np.zeros(2), X_SHOP)
    shape(info, (2, 2), label="glm_fisher_information output")
    close(info, np.array([[3.0, 3.0], [3.0, 5.0]]), atol=1e-12, label="shop information")
    close(info, info.T, atol=1e-12, label="information should be symmetric")


@check("glm_fisher_information matches a finite-difference bend of the gradient")
def test_info_fd(sub):
    for family in ("gaussian", "bernoulli", "poisson"):
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


@check("newton_step returns [1/6, -0.5] on the shop data")
def test_step_shop(sub):
    delta = sub.newton_step("poisson", np.zeros(2), X_SHOP, Y_SHOP)
    shape(delta, (2,), label="newton_step output")
    close(delta, np.array([1.0 / 6.0, -0.5]), atol=1e-10, label="shop step delta")


@check("newton_step gives the hand answers for coins and for Gaussian rows")
def test_step_cases(sub):
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    close(sub.newton_step("bernoulli", np.zeros(2), X, np.array([1.0, 0.0])),
          np.array([-2.0, 2.0]), atol=1e-10, label="Bernoulli step delta")
    close(sub.newton_step("gaussian", np.zeros(2), X, np.array([3.0, 1.0])),
          np.array([-3.0, 1.0]), atol=1e-10, label="Gaussian step delta")


@check("fit_fisher_scoring reproduces the shop iterates")
def test_fit_shop(sub):
    w, history = sub.fit_fisher_scoring("poisson", X_SHOP, Y_SHOP, max_iter=30)
    close(np.array(history[0]), np.array(3.0), atol=1e-12, label="history[0]")
    close(np.array(history[1]), np.array(2.7097367), atol=1e-6, label="history[1]")
    close(np.array(history[2]), np.array(2.6599146), atol=1e-6, label="history[2]")
    close(w, np.array([-0.14507, 0.38416]), atol=1e-4, label="final shop weights")


@check("fit_fisher_scoring recovers known Poisson weights in under ten iterations")
def test_fit_poisson(sub):
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
