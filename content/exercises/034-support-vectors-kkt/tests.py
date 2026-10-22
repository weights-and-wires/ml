"""Checks for the support vectors and KKT checklist exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import almost, check, close, equal, fail, has, shape


def _six():
    x = np.array([-3.0, -1.0, 0.5, 1.0, 3.0, 2.0])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    alpha = np.array([0.0, 1 / 18, 1.0, 1.0, 0.0, 1 / 18])
    return np.outer(x, x), y, alpha, -1 / 3, 1.0


def _solve_dual(K, y, C):
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


@check("margin_groups sorts pushes into 0, 1, and 2, and treats 1e-16 as zero")
def test_groups(sub):
    got = has(sub, "margin_groups")
    alpha = np.array([0.0, 1e-16, 0.5, 1.0, 1.0 - 1e-12])
    g = got(alpha, 1.0, 1e-8)
    shape(g, (5,), label="groups")
    equal(np.asarray(g).astype(int), np.array([0, 0, 1, 2, 2]), label="groups for [0, 1e-16, 0.5, 1, 1 - 1e-12] with C = 1")


@check("margin_groups uses C, not 1, for the upper end")
def test_groups_c(sub):
    got = has(sub, "margin_groups")
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


@check("kkt_violation matches a house-by-house loop on random pushes, with some at 0 and at C")
def test_violation_random(sub):
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


@check("kkt_violation is tiny at an independently computed optimum")
def test_violation_optimum(sub):
    got = has(sub, "kkt_violation")
    X, y = _nine()
    K = X @ X.T
    C = 0.3
    alpha, b = _solve_dual(K, y, C)
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
    alpha, b = _solve_dual(X @ X.T, y, C)
    g = np.asarray(groups(alpha, C, 1e-6)).astype(int)
    keep = g != 0
    if keep.sum() == len(y) or keep.sum() < 2:
        fail(f"expected some, but not all, houses to push; found {int(keep.sum())} of {len(y)}")
    a2, b2 = _solve_dual(X[keep] @ X[keep].T, y[keep], C)
    w_all = (alpha * y) @ X
    w_keep = (a2 * y[keep]) @ X[keep]
    close(w_keep, w_all, atol=1e-4, label="w after deleting the zero-push houses")
