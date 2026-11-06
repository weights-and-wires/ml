"""Checks for the dual and support vectors exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import almost, check, close, equal, fail, has, shape


def _pair():
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, -1.0])
    return X, y


def _ring(n=20, seed=4):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.0, 1.0, n)
    inner = np.c_[r * np.cos(th), r * np.sin(th)]
    th2 = rng.uniform(0, 2 * np.pi, n)
    r2 = rng.uniform(2.0, 2.6, n)
    outer = np.c_[r2 * np.cos(th2), r2 * np.sin(th2)]
    return np.vstack([inner, outer]), np.array([-1.0] * n + [1.0] * n)


def _solve_dual(K, y, C):
    """An independent dual solution, used only to check the reader's scores."""
    n = len(y)
    result = minimize(
        lambda a: -np.sum(a) + 0.5 * (a * y) @ K @ (a * y),
        np.zeros(n),
        jac=lambda a: -(np.ones(n) - y * (K @ (a * y))),
        bounds=[(0.0, C)] * n,
        constraints=[{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


@check("gram_matrix with the linear kind gives the table [[1, -1], [-1, 1]] for two houses at +1 and -1")
def test_gram_linear(sub):
    got = has(sub, "gram_matrix")
    X, _ = _pair()
    K = got(X, "linear")
    shape(K, (2, 2), label="K")
    close(K, np.array([[1.0, -1.0], [-1.0, 1.0]]), label="K for the two-house example")


@check("gram_matrix gives poly and rbf entries that match direct formulas, and is symmetric on random points")
def test_gram_kinds(sub):
    got = has(sub, "gram_matrix")
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    close(got(X, "poly")[0, 1], 144.0, label="poly entry (1 + 11)^2")
    Z = np.array([[0.0, 0.0], [3.0, 4.0]])
    K = got(Z, "rbf", 0.1)
    close(K[0, 1], np.exp(-2.5), label="rbf entry exp(-0.1 * 25)")
    close(np.diag(K), np.ones(2), atol=1e-12, label="rbf diagonal")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 3))
    gamma = 0.7
    K = got(X, "rbf", gamma)
    want = np.array([[np.exp(-gamma * np.sum((p - q) ** 2)) for q in X] for p in X])
    close(K, want, atol=1e-10, label="rbf Gram matrix against a double loop")
    close(K, K.T, atol=1e-12, label="symmetry")


@check("dual_objective is 0.5 on the two-house example and peaks at alpha = 0.5")
def test_dual_objective(sub):
    got = has(sub, "dual_objective")
    X, y = _pair()
    K = X @ X.T
    almost(got(np.array([0.5, 0.5]), y, K), 0.5, 1e-9, label="D at alpha = [0.5, 0.5] (dropping the labels gives 1)")
    for a in (0.25, 0.75):
        want = 2 * a - 2 * a * a
        almost(got(np.array([a, a]), y, K), want, 1e-9, label=f"D at alpha = [{a}, {a}]")


@check("dual_scores equals X @ w + b for a linear K, and gives the two-house pushes a margin of exactly 1")
def test_dual_scores(sub):
    got = has(sub, "dual_scores")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))
    y = np.sign(rng.normal(size=12))
    alpha = rng.uniform(0, 1, size=12)
    b = 0.4
    f = got(alpha, y, X @ X.T, b)
    shape(f, (12,), label="scores")
    close(f, X @ ((alpha * y) @ X) + b, atol=1e-10, label="dual scores against primal scores")

    X, y = _pair()
    f = got(np.array([0.5, 0.5]), y, X @ X.T, 0.0)
    close(y * f, np.ones(2), label="y * f for alpha = [0.5, 0.5], b = 0")


@check("The same dual solver, given an RBF Gram matrix, separates a ring from its centre; a linear one cannot")
def test_ring(sub):
    gram = has(sub, "gram_matrix")
    scores = has(sub, "dual_scores")
    X, y = _ring()
    K_rbf = gram(X, "rbf", 0.5)
    alpha = _solve_dual(K_rbf, y, 10.0)
    free = (alpha > 1e-6) & (alpha < 10.0 - 1e-6)
    f0 = scores(alpha, y, K_rbf, 0.0)
    b = float(np.mean(y[free] - f0[free])) if free.any() else 0.0
    wrong = int(np.sum(np.sign(scores(alpha, y, K_rbf, b)) != y))
    if wrong > 0:
        fail(f"the RBF dual leaves {wrong} of 40 points on the wrong side; expected 0")
    K_lin = gram(X, "linear")
    alpha_l = _solve_dual(K_lin, y, 1.0)
    free_l = (alpha_l > 1e-6) & (alpha_l < 1.0 - 1e-6)
    f0l = scores(alpha_l, y, K_lin, 0.0)
    b_l = float(np.mean(y[free_l] - f0l[free_l])) if free_l.any() else 0.0
    wrong_l = int(np.sum(np.sign(scores(alpha_l, y, K_lin, b_l)) != y))
    if wrong_l < 5:
        fail(f"a linear K should leave many ring points wrong, but only {wrong_l} are")




def _six():
    x = np.array([-3.0, -1.0, 0.5, 1.0, 3.0, 2.0])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    alpha = np.array([0.0, 1 / 18, 1.0, 1.0, 0.0, 1 / 18])
    return np.outer(x, x), y, alpha, -1 / 3, 1.0


def _solve_dual_b(K, y, C):
    """An independent dual solution, used only to check the reader's functions."""
    n = len(y)
    result = minimize(
        lambda a: -np.sum(a) + 0.5 * (a * y) @ K @ (a * y),
        np.zeros(n),
        jac=lambda a: -(np.ones(n) - y * (K @ (a * y))),
        bounds=[(0.0, C)] * n,
        constraints=[{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    alpha = result.x
    f0 = (alpha * y) @ K
    free = (alpha > 1e-6) & (alpha < C - 1e-6)
    b = float(np.mean(y[free] - f0[free])) if free.any() else 0.0
    return alpha, b


def _nine():
    X = np.array(
        [[2, 4.5], [2.5, 2.5], [3, 3.6], [3.4, 1.8], [7, 4.2], [7.5, 2.4], [6.8, 3.3], [8, 1.6], [5.9, 3.0]]
    )
    y = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    return X, y


@check("margin_groups sorts pushes into 0, 1, and 2, treats 1e-16 as zero, and uses C for the upper end")
def test_groups(sub):
    got = has(sub, "margin_groups")
    alpha = np.array([0.0, 1e-16, 0.5, 1.0, 1.0 - 1e-12])
    g = got(alpha, 1.0, 1e-8)
    shape(g, (5,), label="groups")
    equal(np.asarray(g).astype(int), np.array([0, 0, 1, 2, 2]), label="groups for [0, 1e-16, 0.5, 1, 1 - 1e-12] with C = 1")
    g = got(np.array([0.3, 0.05, 0.0, 0.2999999999]), 0.3, 1e-8)
    equal(np.asarray(g).astype(int), np.array([2, 1, 0, 2]), label="groups with C = 0.3")


@check("slack_amounts is [0, 0, 1, 2/3, 0, 0] at the best pushes of the six-house example")
def test_slack(sub):
    got = has(sub, "slack_amounts")
    K, y, alpha, b, _ = _six()
    s = got(alpha, y, K, b)
    shape(s, (6,), label="slack")
    close(s, np.array([0.0, 0.0, 1.0, 2 / 3, 0.0, 0.0]), atol=1e-9, label="slack at the best pushes")
    close(got(np.zeros(6), y, K, 0.0), np.ones(6), label="slack at alpha = 0, b = 0")


@check("kkt_violation is 0 at the best pushes of the six-house example and 1 at alpha = 0")
def test_violation_worked(sub):
    got = has(sub, "kkt_violation")
    K, y, alpha, b, C = _six()
    almost(got(alpha, y, K, b, C), 0.0, 1e-9, label="violation at the best pushes")
    almost(got(np.zeros(6), y, K, 0.0, C), 1.0, 1e-9, label="violation at alpha = 0, b = 0")


@check("kkt_violation matches a house-by-house loop on random pushes, and is tiny at an independently computed optimum")
def test_violation_random_and_optimum(sub):
    got = has(sub, "kkt_violation")
    rng = np.random.default_rng(5)
    n, C = 14, 0.8
    X = rng.normal(size=(n, 2))
    y = np.sign(rng.normal(size=n))
    K = X @ X.T
    alpha = rng.uniform(0, C, size=n)
    alpha[:4] = 0.0
    alpha[4:7] = C
    b = 0.2
    f = (alpha * y) @ K + b
    want = 0.0
    for i in range(n):
        g = y[i] * f[i] - 1.0
        if alpha[i] <= 1e-8:
            v = max(0.0, -g)
        elif alpha[i] >= C - 1e-8:
            v = max(0.0, g)
        else:
            v = abs(g)
        want = max(want, v)
    almost(got(alpha, y, K, b, C), want, 1e-9, label="largest violation")

    X, y = _nine()
    K = X @ X.T
    C = 0.3
    alpha, b = _solve_dual_b(K, y, C)
    v = got(alpha, y, K, b, C, 1e-6)
    if v > 1e-3:
        fail(f"the violation at an optimum should be near 0 but it is {v:.4g}")
    bad = alpha.copy()
    bad[2] = 0.0
    if got(bad, y, K, b, C, 1e-6) < 0.01:
        fail("removing the push of an edge house should create a visible violation")


@check("Houses with no push can be deleted without moving the road")
def test_delete_non_support(sub):
    groups = has(sub, "margin_groups")
    X, y = _nine()
    C = 0.3
    alpha, b = _solve_dual_b(X @ X.T, y, C)
    g = np.asarray(groups(alpha, C, 1e-6)).astype(int)
    keep = g != 0
    if keep.sum() == len(y) or keep.sum() < 2:
        fail(f"expected some, but not all, houses to push; found {int(keep.sum())} of {len(y)}")
    a2, b2 = _solve_dual_b(X[keep] @ X[keep].T, y[keep], C)
    w_all = (alpha * y) @ X
    w_keep = (a2 * y[keep]) @ X[keep]
    close(w_keep, w_all, atol=1e-4, label="w after deleting the zero-push houses")
