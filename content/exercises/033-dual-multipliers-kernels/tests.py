"""Checks for the dual and kernels exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import almost, check, close, fail, has, shape


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


@check("gram_matrix with the poly and rbf kinds matches direct formulas")
def test_gram_kinds(sub):
    got = has(sub, "gram_matrix")
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    close(got(X, "poly")[0, 1], 144.0, label="poly entry (1 + 11)^2")
    Z = np.array([[0.0, 0.0], [3.0, 4.0]])
    K = got(Z, "rbf", 0.1)
    close(K[0, 1], np.exp(-2.5), label="rbf entry exp(-0.1 * 25)")
    close(np.diag(K), np.ones(2), atol=1e-12, label="rbf diagonal")


@check("gram_matrix is symmetric and its RBF entries match a loop on random points")
def test_gram_random(sub):
    got = has(sub, "gram_matrix")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 3))
    gamma = 0.7
    K = got(X, "rbf", gamma)
    want = np.array([[np.exp(-gamma * np.sum((a - b) ** 2)) for b in X] for a in X])
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


@check("dual_scores equals X @ w + b with w = (alpha * y) @ X when K is linear")
def test_dual_scores_linear(sub):
    got = has(sub, "dual_scores")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))
    y = np.sign(rng.normal(size=12))
    alpha = rng.uniform(0, 1, size=12)
    b = 0.4
    f = got(alpha, y, X @ X.T, b)
    shape(f, (12,), label="scores")
    close(f, X @ ((alpha * y) @ X) + b, atol=1e-10, label="dual scores against primal scores")


@check("The two-house pushes give both houses a margin of exactly 1")
def test_dual_margins(sub):
    got = has(sub, "dual_scores")
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
