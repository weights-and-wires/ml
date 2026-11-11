"""Checks for the convex duality exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has


# Independent FISTA for the lasso, so the checks do not rely on the reader's file.
def _fista(X, y, lam, iters):
    step = 1.0 / np.linalg.eigvalsh(X.T @ X).max()
    w = np.zeros(X.shape[1])
    z, t = w.copy(), 1.0
    out = [w.copy()]
    for _ in range(iters):
        u = z - step * X.T @ (X @ z - y)
        w_new = np.sign(u) * np.maximum(np.abs(u) - step * lam, 0.0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + (t - 1.0) / t_new * (w_new - w)
        w, t = w_new, t_new
        out.append(w.copy())
    return out


def _lasso_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 20))
    wt = np.zeros(20)
    wt[:4] = [3.0, -2.0, 1.5, 2.0]
    y = X @ wt + 0.5 * rng.normal(size=100)
    return X, y


def _qp_problem(seed):
    rng = np.random.default_rng(seed)
    d, m = 6, 2
    B = rng.normal(size=(d, d))
    Q = B @ B.T + np.eye(d)
    c = rng.normal(size=d)
    A = rng.normal(size=(m, d))
    b = rng.normal(size=m)
    return Q, c, A, b


@check("lagrangian prices a broken rule: x^2 + 2(1 - x) is 1 at x = 1 and 5 at x = 3")
def test_lagrangian_values(sub):
    fn = has(sub, "lagrangian")
    f = lambda x: x**2
    g = lambda x: 1.0 - x
    almost(fn(1.0, [2.0], [], f, [g], []), 1.0, 1e-12, label="L(1, 2)")
    almost(fn(3.0, [2.0], [], f, [g], []), 5.0, 1e-12, label="L(3, 2)")
    h = lambda x: x - 2.0
    almost(fn(3.0, [2.0], [-1.5], f, [g], [h]), 5.0 - 1.5, 1e-12, label="L(3, 2, -1.5) with h(x) = x - 2")


@check("minimising lagrangian over a grid of x gives the dual function lam - lam^2/4, at most the best cost 1")
def test_one_variable_dual(sub):
    fn = has(sub, "lagrangian")
    f = lambda x: x**2
    g = lambda x: 1.0 - x
    xs = np.linspace(-1.0, 4.0, 5001)
    for lam in [0.0, 1.0, 2.0, 3.0, 4.0]:
        low = min(fn(x, [lam], [], f, [g], []) for x in xs)
        almost(low, lam - lam**2 / 4.0, 1e-5, label=f"dual value at price {lam}")
        if low > 1.0 + 1e-9:
            fail(f"dual value {low} at price {lam} exceeds the best cost 1, so it is not a lower bound")


@check("kkt_residuals is zero at the answer of min 0.5|x|^2 s.t. x1 + x2 >= 1 and positive at a wrong point or price")
def test_kkt(sub):
    fn = has(sub, "kkt_residuals")
    Q, c = np.eye(2), np.zeros(2)
    G, h = np.array([[-1.0, -1.0]]), np.array([-1.0])
    good = fn(Q, c, G, h, np.array([0.5, 0.5]), np.array([0.5]))
    for key in ("stationarity", "primal", "dual", "complementary"):
        almost(good[key], 0.0, 1e-12, label=f"{key} at the answer")
    off = fn(Q, c, G, h, np.array([1.0, 1.0]), np.array([0.5]))
    almost(off["complementary"], 0.5, 1e-12, label="complementary at the loose point (0.5 * 1)")
    almost(off["stationarity"], 0.5, 1e-12, label="stationarity at (1, 1) with price 0.5")
    infeasible = fn(Q, c, G, h, np.array([0.0, 0.0]), np.array([0.5]))
    almost(infeasible["primal"], 1.0, 1e-12, label="primal breach at the origin")
    negative = fn(Q, c, G, h, np.array([0.5, 0.5]), np.array([-0.5]))
    almost(negative["dual"], 0.5, 1e-12, label="dual breach at a negative price")


@check("qp_dual_function matches the Lagrangian minimum: g(nu) = -nu^2 - nu on the small example")
def test_qp_dual_function(sub):
    fn = has(sub, "qp_dual_function")
    Q, c = np.eye(2), np.zeros(2)
    A, b = np.array([[1.0, 1.0]]), np.array([1.0])
    for nu, want in [(-1.5, -0.75), (-1.0, 0.0), (-0.5, 0.25), (0.0, 0.0)]:
        almost(fn(Q, c, A, b, np.array([nu])), want, 1e-12, label=f"g({nu})")
    Q, c, A, b = _qp_problem(1)
    rng = np.random.default_rng(2)
    nu = rng.normal(size=2)
    xs = -np.linalg.solve(Q, c + A.T @ nu)
    lag = 0.5 * xs @ Q @ xs + c @ xs + nu @ (A @ xs - b)
    almost(fn(Q, c, A, b, nu), lag, 1e-9, label="g(nu) on a random problem")


@check("qp_dual returns nu = -0.5, dual value 0.25, and the plan (0.5, 0.5) on the small example")
def test_qp_dual_small(sub):
    fn = has(sub, "qp_dual")
    nu, d, x = fn(np.eye(2), np.zeros(2), np.array([[1.0, 1.0]]), np.array([1.0]))
    close(np.atleast_1d(nu), np.array([-0.5]), atol=1e-12, label="nu*")
    almost(d, 0.25, 1e-12, label="dual value")
    close(x, np.array([0.5, 0.5]), atol=1e-12, label="plan x")


@check("on a random quadratic, the dual optimum equals the primal optimum and bounds it from below at random prices")
def test_qp_dual_random(sub):
    dual = has(sub, "qp_dual")
    dual_fn = has(sub, "qp_dual_function")
    for seed in range(3):
        Q, c, A, b = _qp_problem(seed)
        d_, m_ = Q.shape[0], A.shape[0]
        K = np.block([[Q, A.T], [A, np.zeros((m_, m_))]])
        sol = np.linalg.solve(K, np.concatenate([-c, b]))
        xp = sol[:d_]
        primal = 0.5 * xp @ Q @ xp + c @ xp
        nu, d_star, x = dual(Q, c, A, b)
        almost(d_star, primal, 1e-8, label=f"dual optimum vs primal optimum, seed {seed}")
        close(A @ x, b, atol=1e-8, label=f"plan satisfies A x = b, seed {seed}")
        rng = np.random.default_rng(10 + seed)
        for _ in range(50):
            g = dual_fn(Q, c, A, b, rng.normal(size=m_) * 3.0)
            if g > primal + 1e-9:
                fail(f"dual value {g} exceeds the primal optimum {primal}; weak duality says it cannot")


@check("ridge_dual gives a = [1/6, 1/3], w = 5/6 on two houses, and matches primal ridge on a wide design")
def test_ridge_dual(sub):
    fn = has(sub, "ridge_dual")
    a, w = fn(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), 1.0)
    close(a, np.array([1.0 / 6.0, 1.0 / 3.0]), atol=1e-12, label="a")
    close(w, np.array([5.0 / 6.0]), atol=1e-12, label="w")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 30))
    y = rng.normal(size=8)
    lam = 2.5
    a, w = fn(X, y, lam)
    primal = np.linalg.solve(X.T @ X + lam * np.eye(30), X.T @ y)
    close(w, primal, atol=1e-8, label="dual weights vs primal ridge weights")
    close(X.T @ a, w, atol=1e-10, label="w = X^T a")


# Wrong implementation this suite catches: a gap that uses the raw residual as the dual
# point and skips the rescale into |X^T theta|_inf <= lam. It returns -0.8 at w = 0.4 and
# 0 at w = 0; the hand-value and non-negativity checks below fail it.
@check("lasso_duality_gap is 1.6, 0.4, and 0 at w = 0, 0.4, 0.8 on the two-house example")
def test_gap_hand(sub):
    fn = has(sub, "lasso_duality_gap")
    X, y = np.array([[1.0], [2.0]]), np.array([1.0, 2.0])
    for wv, want in [(0.0, 1.6), (0.4, 0.4), (0.8, 0.0)]:
        got = fn(X, y, np.array([wv]), 1.0)
        almost(got, want, 1e-9, label=f"gap at w = {wv}")


@check("lasso_duality_gap is never negative at random weights, and zero at a long-run lasso answer")
def test_gap_properties(sub):
    fn = has(sub, "lasso_duality_gap")
    X, y = _lasso_data()
    lam = 10.0
    rng = np.random.default_rng(3)
    for _ in range(40):
        w = rng.normal(size=20) * rng.choice([0.0, 0.1, 1.0, 3.0])
        gap = fn(X, y, w, lam)
        if gap < -1e-9:
            fail(f"gap {gap} is negative; the dual point must be scaled into |X^T theta|_inf <= lam")
    w_star = _fista(X, y, lam, 400)[-1]
    almost(fn(X, y, w_star, lam), 0.0, 1e-6, label="gap at the lasso answer")


@check("lasso_duality_gap shrinks over a FISTA run and bounds the true suboptimality")
def test_gap_shrinks(sub):
    fn = has(sub, "lasso_duality_gap")
    X, y = _lasso_data()
    lam = 10.0
    path = _fista(X, y, lam, 400)
    obj = lambda w: 0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w))
    p_star = obj(path[-1])
    gaps = {k: fn(X, y, path[k], lam) for k in (0, 5, 10, 20, 40)}
    ks = sorted(gaps)
    for a, b in zip(ks, ks[1:]):
        if not gaps[b] < gaps[a]:
            fail(f"gap did not shrink from step {a} to step {b}: {gaps[a]} then {gaps[b]}")
    if gaps[40] > 1e-2:
        fail(f"expected a gap under 0.01 after 40 steps, got {gaps[40]}")
    for k, g in gaps.items():
        if g < obj(path[k]) - p_star - 1e-6:
            fail(f"gap {g} at step {k} is below the true suboptimality {obj(path[k]) - p_star}")
