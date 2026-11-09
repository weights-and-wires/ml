"""Checks for the support vector machine exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape


def _houses():
    X = np.array(
        [
            [3.0, 4.5], [2.7, 3.4], [2.5, 3.3], [3.1, 5.4], [2.5, 3.6], [3.5, 4.5],
            [8.1, 1.6], [8.0, 3.0], [6.7, 2.0], [6.1, 1.2], [6.2, 2.2], [6.7, 2.6],
        ]
    )
    y = np.array([-1.0] * 6 + [1.0] * 6)
    return X, y


def _worked():
    X = np.array([[2.0, 0.0], [0.5, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, 1.0, -1.0])
    return np.array([1.0, 0.0]), 0.0, X, y


def _pair():
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, -1.0])
    return X, y, X @ X.T


def _six():
    x = np.array([-3.0, -1.0, 0.5, 1.0, 3.0, 2.0])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    alpha = np.array([0.0, 1 / 18, 1.0, 1.0, 0.0, 1 / 18])
    return np.outer(x, x), y, alpha, -1 / 3, 1.0


def _nine():
    X = np.array(
        [[2, 4.5], [2.5, 2.5], [3, 3.6], [3.4, 1.8], [7, 4.2], [7.5, 2.4], [6.8, 3.3], [8, 1.6], [5.9, 3.0]]
    )
    y = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    return X, y


def _cluster_problem(n=16, seed=0, sep=2.0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack(
        [
            np.array([-sep, 0.0]) + 0.5 * rng.normal(size=(half, 2)),
            np.array([sep, 0.0]) + 0.5 * rng.normal(size=(half, 2)),
        ]
    )
    y = np.concatenate([-np.ones(half), np.ones(half)])
    return X, y, X @ X.T


def _objective(w, b, X, y, C):
    z = y * (X @ w + b)
    return 0.5 * float(w @ w) + C * float(np.sum(np.maximum(0.0, 1.0 - z)))


def _dual(alpha, y, K):
    v = alpha * y
    return float(np.sum(alpha) - 0.5 * v @ K @ v)


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
    return alpha, b, -result.fun


@check("functional_margins gives 2, 0.5 and 1 on the worked example, and flips sign with the label")
def test_margins(sub):
    got = has(sub, "functional_margins")
    w, b, X, y = _worked()
    z = got(w, b, X, y)
    shape(z, (3,), label="z")
    close(z, np.array([2.0, 0.5, 1.0]), label="z on the worked example")
    X2 = np.array([[1.0, 1.0], [1.0, 1.0]])
    z2 = got(np.array([2.0, 1.0]), -1.0, X2, np.array([1.0, -1.0]))
    close(z2, np.array([2.0, -2.0]), label="z for the same point with labels +1 and -1")


@check("hinge_loss_and_subgradient returns J = 1 and the slope [0.5, 0], -1, and matches finite differences")
def test_hinge(sub):
    got = has(sub, "hinge_loss_and_subgradient")
    w, b, X, y = _worked()
    loss, grad_w, grad_b = got(w, b, X, y, 1.0)
    almost(loss, 1.0, 1e-9, label="loss (a mean instead of a sum gives 0.667)")
    shape(grad_w, (2,), label="grad_w")
    close(grad_w, np.array([0.5, 0.0]), label="grad_w (a point at z = 1 must stay inactive)")
    almost(grad_b, -1.0, 1e-9, label="grad_b")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = np.sign(rng.normal(size=20))
    w = rng.normal(size=3)
    b, C = 0.3, 1.5
    if np.any(np.abs(y * (X @ w + b) - 1.0) < 0.05):
        fail("the test data has a margin too close to the corner; change the seed")

    def objective(theta):
        return got(theta[:-1], theta[-1], X, y, C)[0]

    fd = finite_difference(objective, np.concatenate([w, [b]]))
    _, grad_w, grad_b = got(w, b, X, y, C)
    close(np.concatenate([grad_w, [grad_b]]), fd, rtol=1e-4, atol=1e-4, label="subgradient against finite differences")


@check("fit_subgradient puts every house on its correct side and finds a road about as wide as the best one")
def test_fit_subgradient(sub):
    got = has(sub, "fit_subgradient")
    X, y = _houses()
    C = 1.0
    w, b = got(X, y, C, 0.3, 3000)
    shape(w, (2,), label="w")
    z = y * (X @ w + b)
    if z.min() <= 0:
        fail(f"a house is on the wrong side: the smallest z is {z.min():.3f}")
    start = _objective(np.zeros(2), 0.0, X, y, C)
    end = _objective(w, float(b), X, y, C)
    if end > 0.05 * start:
        fail(f"the objective only fell from {start:.3f} to {end:.3f}; expected below {0.05 * start:.3f}")
    _, _, best = _solve_dual(X @ X.T, y, C)
    if end > 1.05 * best:
        fail(f"objective {end:.4f} is more than 5% above the best value {best:.4f}")
    half = 1.0 / np.linalg.norm(w)
    best_half = 1.0 / np.sqrt(2.0 * best)
    almost(half, best_half, 0.1 * best_half, label="road half-width 1/||w||")


@check("dual_objective is 0.5 at alpha = [0.5, 0.5], and dual_scores equals X @ w + b for a dot-product K")
def test_dual(sub):
    objective = has(sub, "dual_objective")
    scores = has(sub, "dual_scores")
    X, y, K = _pair()
    almost(objective(np.array([0.5, 0.5]), y, K), 0.5, 1e-9, label="D at alpha = [0.5, 0.5] (dropping the labels gives 1)")
    for a in (0.25, 0.75):
        almost(objective(np.array([a, a]), y, K), 2 * a - 2 * a * a, 1e-9, label=f"D at alpha = [{a}, {a}]")
    f = scores(np.array([0.5, 0.5]), y, K, 0.0)
    close(y * f, np.ones(2), label="y * f for alpha = [0.5, 0.5], b = 0")

    rng = np.random.default_rng(2)
    Xr = rng.normal(size=(12, 3))
    yr = np.sign(rng.normal(size=12))
    alpha = rng.uniform(0, 1, size=12)
    f = scores(alpha, yr, Xr @ Xr.T, 0.4)
    shape(f, (12,), label="scores")
    close(f, Xr @ ((alpha * yr) @ Xr) + 0.4, atol=1e-10, label="dual scores against primal scores")


@check("margin_groups sorts pushes into 0, 1, and 2, treats 1e-16 as zero, and uses C for the upper end")
def test_groups(sub):
    got = has(sub, "margin_groups")
    alpha = np.array([0.0, 1e-16, 0.5, 1.0, 1.0 - 1e-12])
    g = got(alpha, 1.0, 1e-8)
    shape(g, (5,), label="groups")
    equal(np.asarray(g).astype(int), np.array([0, 0, 1, 2, 2]), label="groups for [0, 1e-16, 0.5, 1, 1 - 1e-12] with C = 1")
    g = got(np.array([0.3, 0.05, 0.0, 0.2999999999]), 0.3, 1e-8)
    equal(np.asarray(g).astype(int), np.array([2, 1, 0, 2]), label="groups with C = 0.3")


@check("kkt_violation is 0 at the best pushes of the six-house example, 1 at alpha = 0, and tiny at a true optimum")
def test_violation(sub):
    got = has(sub, "kkt_violation")
    K, y, alpha, b, C = _six()
    almost(got(alpha, y, K, b, C), 0.0, 1e-9, label="violation at the best pushes")
    almost(got(np.zeros(6), y, K, 0.0, C), 1.0, 1e-9, label="violation at alpha = 0, b = 0")

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
    almost(got(alpha, y, K, b, C), want, 1e-9, label="largest violation against a house-by-house loop")

    X, y = _nine()
    K = X @ X.T
    C = 0.3
    alpha, b, _ = _solve_dual(K, y, C)
    v = got(alpha, y, K, b, C, 1e-6)
    if v > 1e-3:
        fail(f"the violation at an optimum should be near 0 but it is {v:.4g}")
    bad = alpha.copy()
    bad[2] = 0.0
    if got(bad, y, K, b, C, 1e-6) < 0.01:
        fail("removing the push of an edge house should create a visible violation")


@check("clip_bounds gives (0, 10) and (2, 10) on the two cases of the chapter, and respects the order of i and j")
def test_clip_bounds(sub):
    got = has(sub, "clip_bounds")
    low, high = got(np.zeros(2), np.array([1.0, -1.0]), 10.0, 0, 1)
    almost(low, 0.0, 1e-12, label="L when the labels differ and both pushes are 0")
    almost(high, 10.0, 1e-12, label="H when the labels differ and both pushes are 0")
    low, high = got(np.array([7.0, 5.0]), np.array([1.0, 1.0]), 10.0, 0, 1)
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


@check("smo_step solves the two-house example in one step, stops at the wall, and returns the inputs when it cannot move")
def test_smo_step(sub):
    got = has(sub, "smo_step")
    _, y, K = _pair()
    alpha0 = np.zeros(2)
    alpha, b = got(alpha0, y, K, 0.0, 10.0, 0, 1)
    shape(alpha, (2,), label="alpha")
    close(alpha, np.array([0.5, 0.5]), atol=1e-9, label="alpha after one step")
    almost(b, 0.0, 1e-9, label="b after one step")
    almost(_dual(alpha, y, K), 0.5, 1e-9, label="D after one step")
    close(alpha0, np.zeros(2), label="the input alpha, which must not be changed")

    alpha, b = got(np.zeros(2), y, K, 0.0, 0.3, 0, 1)
    close(alpha, np.array([0.3, 0.3]), atol=1e-9, label="alpha with C = 0.3")
    almost(b, 0.0, 1e-9, label="b when both pushes end at the cap")

    a0 = np.array([0.2, 0.2])
    a, b = got(a0, y, K, 0.1, 10.0, 1, 1)
    close(a, a0, label="alpha when i == j")
    almost(b, 0.1, 1e-12, label="b when i == j")
    a, b = got(a0, y, np.ones((2, 2)), 0.1, 10.0, 0, 1)
    close(a, a0, label="alpha when the two points coincide")
    almost(b, 0.1, 1e-12, label="b when the two points coincide")
    a, b = got(np.array([0.0, 10.0]), y, K, 0.1, 10.0, 0, 1)
    close(a, np.array([0.0, 10.0]), label="alpha when L >= H")
    almost(b, 0.1, 1e-12, label="b when L >= H")


@check("Across many random pairs, alpha stays in [0, C], the pushes balance, D never falls, and inside houses rest on their edge")
def test_smo_feasible_monotone(sub):
    got = has(sub, "smo_step")
    _, y, K = _cluster_problem(n=12, seed=2)
    n, C = len(y), 1.0
    rng = np.random.default_rng(3)
    alpha, b = np.zeros(n), 0.0
    previous = _dual(alpha, y, K)
    checked = 0
    for step in range(80):
        i, j = (int(v) for v in rng.choice(n, size=2, replace=False))
        new_alpha, new_b = got(alpha, y, K, b, C, i, j)
        if np.any(new_alpha < -1e-9) or np.any(new_alpha > C + 1e-9):
            fail(f"after step {step}, alpha leaves [0, {C}]: min {new_alpha.min():.4g}, max {new_alpha.max():.4g}")
        if abs(float(new_alpha @ y)) > 1e-8:
            fail(f"after step {step}, sum_i alpha_i y_i is {float(new_alpha @ y):.3g}, expected 0")
        current = _dual(new_alpha, y, K)
        if current < previous - 1e-9:
            fail(f"after step {step}, D fell from {previous:.6f} to {current:.6f}")
        f = (new_alpha * y) @ K + new_b
        if not np.allclose(new_alpha, alpha):
            for k in (i, j):
                if 1e-9 < new_alpha[k] < C - 1e-9:
                    checked += 1
                    if abs(y[k] * f[k] - 1.0) > 1e-7:
                        fail(f"house {k} has 0 < alpha < C but y f = {y[k] * f[k]:.6f}, expected 1")
                    break
        alpha, b, previous = new_alpha, new_b, current
    if checked == 0:
        fail("the test never saw a house strictly inside the box; change the seed")


@check("fit_smo drives the violation below its tolerance, finds the road of an independent solver, and the zero-push houses can go")
def test_fit_smo(sub):
    got = has(sub, "fit_smo")
    kkt = has(sub, "kkt_violation")
    groups = has(sub, "margin_groups")
    X, y, K = _cluster_problem()
    C, tol = 1.0, 1e-3
    alpha, b = got(K, y, C, 200, tol)
    shape(alpha, (len(y),), label="alpha")
    violation = kkt(alpha, y, K, b, C)
    if violation > 5 * tol:
        fail(f"the largest violation is {violation:.4g}, above {5 * tol:.4g}")
    if np.any(alpha < -1e-9) or np.any(alpha > C + 1e-9):
        fail("alpha leaves the box [0, C]")

    alpha, b = got(K, y, C, 200, 1e-4)
    w = (alpha * y) @ X
    w_ref = (_solve_dual(K, y, C)[0] * y) @ X
    close(w, w_ref, atol=0.02, label="w against the independent solver")

    X9, y9 = _nine()
    a9, b9 = got(X9 @ X9.T, y9, 0.3, 500, 1e-6)
    g = np.asarray(groups(a9, 0.3, 1e-4)).astype(int)
    keep = g != 0
    if keep.sum() == len(y9) or keep.sum() < 2:
        fail(f"expected some, but not all, houses to push; found {int(keep.sum())} of {len(y9)}")
    a2, _ = got(X9[keep] @ X9[keep].T, y9[keep], 0.3, 500, 1e-6)
    close((a2 * y9[keep]) @ X9[keep], (a9 * y9) @ X9, atol=2e-3, label="w after deleting the zero-push houses")
