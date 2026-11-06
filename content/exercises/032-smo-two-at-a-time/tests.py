"""Checks for the SMO exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import almost, check, close, equal, fail, has, shape


def _pair():
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, -1.0])
    return X, y, X @ X.T


def _problem(n=16, seed=0, d=2, sep=2.0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack(
        [
            np.array([-sep, 0.0]) + 0.5 * rng.normal(size=(half, d)),
            np.array([sep, 0.0]) + 0.5 * rng.normal(size=(half, d)),
        ]
    )
    y = np.concatenate([-np.ones(half), np.ones(half)])
    return X, y, X @ X.T


def _dual(alpha, y, K):
    v = alpha * y
    return float(np.sum(alpha) - 0.5 * v @ K @ v)


def _reference_alpha(y, K, C):
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


@check("clip_bounds gives (0, 10) and (2, 10) on the two cases of the chapter")
def test_clip_bounds(sub):
    got = has(sub, "clip_bounds")
    y_diff = np.array([1.0, -1.0])
    low, high = got(np.zeros(2), y_diff, 10.0, 0, 1)
    almost(low, 0.0, 1e-12, label="L when the labels differ and both pushes are 0")
    almost(high, 10.0, 1e-12, label="H when the labels differ and both pushes are 0")
    y_same = np.array([1.0, 1.0])
    low, high = got(np.array([7.0, 5.0]), y_same, 10.0, 0, 1)
    almost(low, 2.0, 1e-12, label="L when the labels match and alpha_i + alpha_j = 12")
    almost(high, 10.0, 1e-12, label="H when the labels match and alpha_i + alpha_j = 12")


    y = np.array([1.0, -1.0])
    alpha = np.array([3.0, 1.0])
    low, high = got(alpha, y, 10.0, 0, 1)
    almost(low, 0.0, 1e-12, label="L for (i, j) = (0, 1)")
    almost(high, 8.0, 1e-12, label="H for (i, j) = (0, 1)")
    low, high = got(alpha, y, 10.0, 1, 0)
    almost(low, 2.0, 1e-12, label="L for (i, j) = (1, 0)")
    almost(high, 10.0, 1e-12, label="H for (i, j) = (1, 0)")


@check("smo_step on the two-house example returns alpha = [0.5, 0.5] and b = 0, and D rises from 0 to 0.5")
def test_worked_step(sub):
    got = has(sub, "smo_step")
    _, y, K = _pair()
    alpha0 = np.zeros(2)
    alpha, b = got(alpha0, y, K, 0.0, 10.0, 0, 1)
    shape(alpha, (2,), label="alpha")
    close(alpha, np.array([0.5, 0.5]), atol=1e-9, label="alpha after one step")
    almost(b, 0.0, 1e-9, label="b after one step")
    almost(_dual(alpha, y, K), 0.5, 1e-9, label="D after one step")
    close(alpha0, np.zeros(2), label="the input alpha, which must not be changed")


@check("smo_step stops at the wall when the peak is outside the box")
def test_wall(sub):
    got = has(sub, "smo_step")
    _, y, K = _pair()
    alpha, b = got(np.zeros(2), y, K, 0.0, 0.3, 0, 1)
    close(alpha, np.array([0.3, 0.3]), atol=1e-9, label="alpha with C = 0.3")
    almost(b, 0.0, 1e-9, label="b when both pushes end at the cap")


@check("smo_step returns the inputs unchanged when i == j, when eta is 0, and when L >= H")
def test_unchanged(sub):
    got = has(sub, "smo_step")
    _, y, K = _pair()
    a0 = np.array([0.2, 0.2])
    a, b = got(a0, y, K, 0.1, 10.0, 1, 1)
    close(a, a0, label="alpha when i == j")
    almost(b, 0.1, 1e-12, label="b when i == j")
    K_same = np.ones((2, 2))
    a, b = got(a0, y, K_same, 0.1, 10.0, 0, 1)
    close(a, a0, label="alpha when the two points coincide")
    almost(b, 0.1, 1e-12, label="b when the two points coincide")
    a, b = got(np.array([0.0, 10.0]), y, K, 0.1, 10.0, 0, 1)
    close(a, np.array([0.0, 10.0]), label="alpha when L >= H")
    almost(b, 0.1, 1e-12, label="b when L >= H")


@check("Across many random pairs, alpha stays in [0, C], the pushes stay balanced, and D never falls")
def test_feasible_monotone(sub):
    got = has(sub, "smo_step")
    _, y, K = _problem(n=12, seed=2)
    n, C = len(y), 1.0
    rng = np.random.default_rng(3)
    alpha, b = np.zeros(n), 0.0
    previous = _dual(alpha, y, K)
    for step in range(80):
        i, j = rng.choice(n, size=2, replace=False)
        alpha, b = got(alpha, y, K, b, C, int(i), int(j))
        if np.any(alpha < -1e-9) or np.any(alpha > C + 1e-9):
            fail(f"after step {step}, alpha leaves [0, {C}]: min {alpha.min():.4g}, max {alpha.max():.4g}")
        if abs(float(alpha @ y)) > 1e-8:
            fail(f"after step {step}, sum_i alpha_i y_i is {float(alpha @ y):.3g}, expected 0")
        current = _dual(alpha, y, K)
        if current < previous - 1e-9:
            fail(f"after step {step}, D fell from {previous:.6f} to {current:.6f}")
        previous = current


@check("After a step, a house strictly inside (0, C) rests exactly on its edge, y f = 1")
def test_bias_rule(sub):
    got = has(sub, "smo_step")
    _, y, K = _problem(n=12, seed=2)
    n, C = len(y), 1.0
    rng = np.random.default_rng(4)
    alpha, b = np.zeros(n), 0.0
    checked = 0
    for _ in range(60):
        i, j = (int(v) for v in rng.choice(n, size=2, replace=False))
        new_alpha, new_b = got(alpha, y, K, b, C, i, j)
        f = (new_alpha * y) @ K + new_b
        changed = not np.allclose(new_alpha, alpha)
        for k in (i, j):
            if changed and 1e-9 < new_alpha[k] < C - 1e-9:
                checked += 1
                if abs(y[k] * f[k] - 1.0) > 1e-7:
                    fail(f"house {k} has 0 < alpha < C but y f = {y[k] * f[k]:.6f}, expected 1")
                break
        alpha, b = new_alpha, new_b
    if checked == 0:
        fail("the test never saw a house strictly inside the box; change the seed")


@check("fit_smo drives the largest KKT violation below its tolerance")
def test_fit_converges(sub):
    got = has(sub, "fit_smo")
    kkt = has(sub, "kkt_violation")
    _, y, K = _problem()
    C, tol = 1.0, 1e-3
    alpha, b = got(K, y, C, 200, tol)
    shape(alpha, (len(y),), label="alpha")
    violation = kkt(alpha, y, K, b, C)
    if violation > 5 * tol:
        fail(f"the largest violation is {violation:.4g}, above {5 * tol:.4g}")
    if np.any(alpha < -1e-9) or np.any(alpha > C + 1e-9):
        fail("alpha leaves the box [0, C]")


@check("fit_smo finds the same road as an independent solver")
def test_fit_matches_reference(sub):
    got = has(sub, "fit_smo")
    X, y, K = _problem()
    C = 1.0
    alpha, b = got(K, y, C, 200, 1e-4)
    w = (alpha * y) @ X
    w_ref = (_reference_alpha(y, K, C) * y) @ X
    close(w, w_ref, atol=0.02, label="w against the independent solver")
