"""Checks for the ridge penalty and closed-form exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, shape


def random_problem(seed: int, n: int = 25, d: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    return X, y


@check("ridge_objective scores the hand example: 2.5 at (1.5, 0.5) and 5 at (3, 1)")
def test_objective_hand(sub):
    X = np.eye(2)
    y = np.array([3.0, 1.0])
    almost(sub.ridge_objective(X, y, np.array([1.5, 0.5]), 1.0), 2.5, 1e-12, label="score at (1.5, 0.5)")
    almost(sub.ridge_objective(X, y, np.array([3.0, 1.0]), 1.0), 5.0, 1e-12, label="score at (3, 1)")


@check("ridge_objective adds exactly half lambda times the squared length to the misses")
def test_objective_fee(sub):
    X, y = random_problem(0)
    rng = np.random.default_rng(1)
    w = rng.normal(size=4)
    base = sub.ridge_objective(X, y, w, 0.0)
    almost(base, 0.5 * np.sum((X @ w - y) ** 2), 1e-10, label="score at lam = 0")
    for lam in (0.5, 3.0):
        fee = sub.ridge_objective(X, y, w, lam) - base
        almost(fee, 0.5 * lam * np.dot(w, w), 1e-9, label=f"fee at lam = {lam}")


@check("ridge_gradient matches finite differences of ridge_objective")
def test_gradient_finite_difference(sub):
    X, y = random_problem(2)
    rng = np.random.default_rng(3)
    for lam in (0.0, 0.7, 5.0):
        w = rng.normal(size=4)
        got = sub.ridge_gradient(X, y, w, lam)
        shape(got, (4,), label="ridge_gradient output")
        want = finite_difference(lambda v: sub.ridge_objective(X, y, v, lam), w.copy())
        close(got, want, rtol=1e-5, atol=1e-5, label=f"gradient against finite differences at lam = {lam}")


@check("ridge_closed_form finds a point where the ridge gradient is zero")
def test_zero_gradient(sub):
    X, y = random_problem(4, n=40, d=6)
    for lam in (0.1, 2.0, 50.0):
        w = sub.ridge_closed_form(X, y, lam)
        shape(w, (6,), label="ridge_closed_form output")
        grad = X.T @ (X @ w - y) + lam * w
        close(grad, np.zeros(6), atol=1e-8, label=f"gradient at the solution, lam = {lam}")


@check("ridge_closed_form agrees with least squares on stacked data, and with lstsq at lam = 0")
def test_against_lstsq(sub):
    X, y = random_problem(5, n=30, d=5)
    lam = 4.0
    stacked_X = np.vstack([X, np.sqrt(lam) * np.eye(5)])
    stacked_y = np.concatenate([y, np.zeros(5)])
    want = np.linalg.lstsq(stacked_X, stacked_y, rcond=None)[0]
    close(sub.ridge_closed_form(X, y, lam), want, atol=1e-8, label="against the stacked least squares solve")
    close(
        sub.ridge_closed_form(X, y, 0.0),
        np.linalg.lstsq(X, y, rcond=None)[0],
        atol=1e-8,
        label="lam = 0 against plain least squares",
    )


@check("ridge_closed_form splits the weight evenly between two identical columns")
def test_duplicate_columns(sub):
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 2.0])
    w = sub.ridge_closed_form(X, y, 1.0)
    if not np.all(np.isfinite(w)):
        fail(f"ridge_closed_form returned {w} on identical columns; X^T X + lam I is solvable when lam > 0")
    close(w, np.array([5.0 / 11.0, 5.0 / 11.0]), atol=1e-10, label="weights on identical columns at lam = 1")


@check("the ridge solution shrinks as lambda grows, and beats nearby weights on the score")
def test_shrinks(sub):
    X, y = random_problem(6, n=40, d=5)
    norms = [np.linalg.norm(sub.ridge_closed_form(X, y, lam)) for lam in (0.0, 0.5, 5.0, 50.0, 500.0)]
    if not all(a > b for a, b in zip(norms, norms[1:])):
        fail(f"the weight length should fall as lam rises, but the lengths were {np.round(norms, 4)}")
    rng = np.random.default_rng(7)
    lam = 3.0
    w = sub.ridge_closed_form(X, y, lam)
    best = sub.ridge_objective(X, y, w, lam)
    for _ in range(50):
        other = w + 0.05 * rng.normal(size=5)
        if sub.ridge_objective(X, y, other, lam) < best - 1e-12:
            fail("a nearby set of weights scores lower than the closed-form solution, so it is not the bottom")
