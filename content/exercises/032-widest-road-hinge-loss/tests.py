"""Checks for the widest road and hinge loss exercise."""

import numpy as np
from scipy.optimize import minimize

from mlcheck import almost, check, close, fail, finite_difference, has, shape


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


def _objective(w, b, X, y, C):
    z = y * (X @ w + b)
    return 0.5 * float(w @ w) + C * float(np.sum(np.maximum(0.0, 1.0 - z)))


def _reference_optimum(X, y, C):
    """The best objective value, from an independent constrained optimiser."""
    K = X @ X.T
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
    return -result.fun


@check("functional_margins gives 2, 0.5 and 1 on the worked example")
def test_margins(sub):
    got = has(sub, "functional_margins")
    w, b, X, y = _worked()
    z = got(w, b, X, y)
    shape(z, (3,), label="z")
    close(z, np.array([2.0, 0.5, 1.0]), label="z on the worked example")


@check("functional_margins is negative for a point on the wrong side, whatever its label")
def test_margins_sign(sub):
    got = has(sub, "functional_margins")
    X = np.array([[1.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, -1.0])
    z = got(np.array([2.0, 1.0]), -1.0, X, y)
    close(z, np.array([2.0, -2.0]), label="z for the same point with labels +1 and -1")


@check("hinge_loss_and_subgradient returns J = 1 and the slope [0.5, 0], -1 on the worked example")
def test_worked_example(sub):
    got = has(sub, "hinge_loss_and_subgradient")
    w, b, X, y = _worked()
    loss, grad_w, grad_b = got(w, b, X, y, 1.0)
    almost(loss, 1.0, 1e-9, label="loss (a mean instead of a sum gives 0.667)")
    shape(grad_w, (2,), label="grad_w")
    close(grad_w, np.array([0.5, 0.0]), label="grad_w (a point at z = 1 must stay inactive)")
    almost(grad_b, -1.0, 1e-9, label="grad_b")


@check("The subgradient matches finite differences away from the corner")
def test_subgradient_fd(sub):
    got = has(sub, "hinge_loss_and_subgradient")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = np.sign(rng.normal(size=20))
    w = rng.normal(size=3)
    b = 0.3
    C = 1.5
    z = y * (X @ w + b)
    if np.any(np.abs(z - 1.0) < 0.05):
        fail("the test data has a margin too close to the corner; change the seed")

    def objective(theta):
        return got(theta[:-1], theta[-1], X, y, C)[0]

    fd = finite_difference(objective, np.concatenate([w, [b]]))
    _, grad_w, grad_b = got(w, b, X, y, C)
    close(np.concatenate([grad_w, [grad_b]]), fd, rtol=1e-4, atol=1e-4, label="subgradient against finite differences")


@check("fit_subgradient puts every house on its correct side and lowers the objective")
def test_fit_separates(sub):
    got = has(sub, "fit_subgradient")
    X, y = _houses()
    w, b = got(X, y, 1.0, 0.3, 3000)
    shape(w, (2,), label="w")
    z = y * (X @ w + b)
    if z.min() <= 0:
        fail(f"a house is on the wrong side: the smallest z is {z.min():.3f}")
    start = _objective(np.zeros(2), 0.0, X, y, 1.0)
    end = _objective(w, float(b), X, y, 1.0)
    if end > 0.05 * start:
        fail(f"the objective only fell from {start:.3f} to {end:.3f}; expected below {0.05 * start:.3f}")


@check("fit_subgradient finds a road about as wide as the best one")
def test_fit_near_optimum(sub):
    got = has(sub, "fit_subgradient")
    X, y = _houses()
    C = 1.0
    w, b = got(X, y, C, 0.3, 3000)
    best = _reference_optimum(X, y, C)
    end = _objective(w, float(b), X, y, C)
    if end > 1.05 * best:
        fail(f"objective {end:.4f} is more than 5% above the best value {best:.4f}")
    half = 1.0 / np.linalg.norm(w)
    best_half = 1.0 / np.sqrt(2.0 * best)
    almost(half, best_half, 0.1 * best_half, label="road half-width 1/||w||")
