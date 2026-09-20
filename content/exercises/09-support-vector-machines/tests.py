"""Checks for the support vector machine exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import check, close, fail, finite_difference, has, shape


def _separable_problem(n: int = 16, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu0, mu1 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    X = np.vstack(
        [mu0 + 0.5 * rng.normal(size=(n // 2, d)), mu1 + 0.5 * rng.normal(size=(n // 2, d))]
    )
    y = np.concatenate([-np.ones(n // 2), np.ones(n // 2)])
    K = X @ X.T
    return X, y, K


def _dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    return float(np.sum(alpha) - 0.5 * (alpha * y) @ K @ (alpha * y))


def _reference_alpha(y: np.ndarray, K: np.ndarray, C: float):
    """An independent dual solution from a generic constrained optimiser, used
    only to check the SMO implementation, never called by the reader's code."""
    n = len(y)

    def neg_dual(a):
        return -np.sum(a) + 0.5 * (a * y) @ K @ (a * y)

    def neg_dual_grad(a):
        return -(np.ones(n) - y * (K @ (a * y)))

    constraints = [{"type": "eq", "fun": lambda a: np.sum(a * y)}]
    result = minimize(
        neg_dual,
        np.zeros(n),
        jac=neg_dual_grad,
        bounds=[(0.0, C)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


@check("hinge_loss_and_subgradient computes the objective value correctly")
def test_hinge_loss_value(sub):
    got = has(sub, "hinge_loss_and_subgradient")
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    y = np.array([1.0, -1.0, 1.0])
    w = np.array([0.5, 0.5])
    b = 0.1
    C = 2.0
    # margins: y_i (w.x_i + b) = 1*(0.6) = 0.6, -1*(-0.4)=0.4, 1*(1.1)=1.1
    want_loss = 0.5 * (0.5**2 + 0.5**2) + C * (max(0, 1 - 0.6) + max(0, 1 - 0.4) + max(0, 1 - 1.1))
    loss, grad_w, grad_b = got(w, b, X, y, C)
    if abs(loss - want_loss) > 1e-8:
        fail(f"loss is {loss:.6f}, expected {want_loss:.6f}")
    shape(grad_w, (2,), label="grad_w")


@check("hinge_loss_and_subgradient's subgradient matches finite differences away from the kink")
def test_hinge_subgradient(sub):
    got = has(sub, "hinge_loss_and_subgradient")
    rng = np.random.default_rng(1)
    n, d = 20, 3
    X = rng.normal(size=(n, d))
    y = np.sign(rng.normal(size=n))
    w = rng.normal(size=d)
    b = 0.3
    C = 1.5

    margins = y * (X @ w + b)
    if np.any(np.abs(margins - 1.0) < 0.05):
        fail("test data has a margin too close to the kink; regenerate it")

    def objective(theta):
        loss, _, _ = got(theta[:-1], theta[-1], X, y, C)
        return loss

    theta = np.concatenate([w, [b]])
    fd_grad = finite_difference(objective, theta)

    _, grad_w, grad_b = got(w, b, X, y, C)
    analytic = np.concatenate([grad_w, [grad_b]])
    close(analytic, fd_grad, rtol=1e-4, atol=1e-4, label="subgradient against finite differences")


@check("kkt_violation is near zero at an independently computed optimum, and large at alpha = 0")
def test_kkt_violation(sub):
    got = has(sub, "kkt_violation")
    _, y, K = _separable_problem()
    C = 1.0

    n = len(y)
    violation_zero = got(np.zeros(n), y, K, 0.0, C)
    if violation_zero < 0.5:
        fail(
            "alpha = 0 badly violates the KKT conditions on separable data (every "
            f"margin point needs alpha > 0), but the reported violation is only {violation_zero:.4f}"
        )

    alpha_ref = _reference_alpha(y, K, C)
    free = (alpha_ref > 1e-4) & (alpha_ref < C - 1e-4)
    # b from any free support vector: y_i (sum_k alpha_k y_k K[i,k] + b) = 1
    scores = (alpha_ref * y) @ K
    b_ref = float(np.mean(y[free] - scores[free])) if free.any() else 0.0
    violation_ref = got(alpha_ref, y, K, b_ref, C)
    if violation_ref > 1e-3:
        fail(
            f"the KKT violation of an independently computed near-optimal solution "
            f"should be tiny, but it is {violation_ref:.4f}"
        )


@check("smo_step keeps alpha inside [0, C] and preserves sum_i alpha_i y_i exactly")
def test_smo_step_feasibility(sub):
    got = has(sub, "smo_step")
    _, y, K = _separable_problem(n=12, seed=2)
    n = len(y)
    C = 1.0
    rng = np.random.default_rng(3)

    alpha = np.zeros(n)
    b = 0.0
    for step in range(60):
        i, j = rng.choice(n, size=2, replace=False)
        alpha, b = got(alpha, y, K, b, C, int(i), int(j))
        if np.any(alpha < -1e-9) or np.any(alpha > C + 1e-9):
            fail(f"after step {step}, alpha leaves [0, {C}]: min {alpha.min():.4g}, max {alpha.max():.4g}")
        equality = float(np.sum(alpha * y))
        if abs(equality) > 1e-8:
            fail(f"after step {step}, sum_i alpha_i y_i is {equality:.3g}, expected 0")


@check("The dual objective increases monotonically across a sequence of SMO steps")
def test_dual_objective_monotone(sub):
    got = has(sub, "smo_step")
    _, y, K = _separable_problem(n=12, seed=2)
    n = len(y)
    C = 1.0

    alpha = np.zeros(n)
    b = 0.0
    previous = _dual_objective(alpha, y, K)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            alpha, b = got(alpha, y, K, b, C, i, j)
            current = _dual_objective(alpha, y, K)
            if current < previous - 1e-9:
                fail(
                    f"dual objective dropped from {previous:.6f} to {current:.6f} "
                    f"after a step on (i={i}, j={j})"
                )
            previous = current


@check("fit_smo drives the KKT violation below its tolerance")
def test_fit_smo_converges(sub):
    got = has(sub, "fit_smo")
    _, y, K = _separable_problem()
    C = 1.0
    tol = 1e-3

    alpha, b = got(K, y, C, max_iter=200, tol=tol)
    shape(alpha, (len(y),), label="alpha")

    kkt = has(sub, "kkt_violation")
    violation = kkt(alpha, y, K, b, C)
    if violation > tol * 5:
        fail(f"fit_smo returned a solution with KKT violation {violation:.4g}, above {tol * 5:.4g}")


@check("fit_smo recovers w that matches a reference solution on a small separable problem")
def test_fit_smo_matches_reference(sub):
    got = has(sub, "fit_smo")
    X, y, K = _separable_problem()
    C = 1.0

    alpha, b = got(K, y, C, max_iter=200, tol=1e-4)
    w = (alpha * y) @ X

    alpha_ref = _reference_alpha(y, K, C)
    w_ref = (alpha_ref * y) @ X
    close(w, w_ref, atol=0.02, label="recovered w against the reference solution")
