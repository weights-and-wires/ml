"""Checks for the walking-downhill exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape

A = np.diag([1.0, 10.0])
B = np.array([1.0, 10.0])


@check("quadratic_loss_and_grad gives the hand-computed heights -5.5, 0, and -5")
def test_hand_losses(sub):
    got = has(sub, "quadratic_loss_and_grad")
    for point, want in [((0.0, 0.0), 0.0), ((1.0, 1.0), -5.5), ((0.0, 1.0), -5.0)]:
        loss, _ = got(A, B, np.array(point))
        if not isinstance(loss, float):
            fail(f"the loss must be a Python float, got {type(loss).__name__}")
        if abs(loss - want) > 1e-9:
            fail(f"loss at {point} is {loss}, expected {want}")


@check("quadratic_loss_and_grad returns the gradient (-1, -10) at (0, 0) and (0, -10) at (1, 0)")
def test_hand_gradients(sub):
    got = has(sub, "quadratic_loss_and_grad")
    _, g0 = got(A, B, np.array([0.0, 0.0]))
    _, g1 = got(A, B, np.array([1.0, 0.0]))
    close(g0, np.array([-1.0, -10.0]), atol=1e-9, label="gradient at (0, 0)")
    close(g1, np.array([0.0, -10.0]), atol=1e-9, label="gradient at (1, 0)")


@check("The gradient matches a finite difference on a random bowl")
def test_gradient_matches_finite_difference(sub):
    got = has(sub, "quadratic_loss_and_grad")
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    Asym = M @ M.T + np.eye(4)
    b = rng.normal(size=4)
    w = rng.normal(size=4)
    _, grad = got(Asym, b, w)
    want = finite_difference(lambda x: 0.5 * x @ Asym @ x - b @ x, w)
    close(grad, want, atol=1e-5, label="analytic vs finite-difference gradient")


@check("numerical_gradient recovers the gradient of a function that is not a bowl")
def test_numerical_gradient(sub):
    got = has(sub, "numerical_gradient")
    f = lambda v: float(np.sin(v[0]) + v[1] ** 3 + v[0] * v[2])
    w = np.array([0.4, -1.2, 2.0])
    before = w.copy()
    grad = got(f, w)
    want = np.array([np.cos(0.4) + 2.0, 3 * 1.44, 0.4])
    close(grad, want, atol=1e-5, label="numerical gradient")
    close(w, before, label="the input w, which numerical_gradient must not change")


@check("gradient_descent returns iters + 1 rows and starts at w0 without changing it")
def test_path_shape(sub):
    got = has(sub, "gradient_descent")
    grad = lambda w: A @ w - B
    w0 = np.array([0.0, 0.0])
    path = got(grad, w0, 0.1, 5)
    shape(path, (6, 2), label="gradient_descent path")
    close(path[0], np.array([0.0, 0.0]), label="row 0")
    close(w0, np.array([0.0, 0.0]), label="the input w0, which gradient_descent must not change")


@check("gradient_descent follows the hand path (0, 0), (0.1, 1.0), (0.19, 1.0), (0.271, 1.0)")
def test_hand_path(sub):
    got = has(sub, "gradient_descent")
    path = got(lambda w: A @ w - B, np.array([0.0, 0.0]), 0.1, 3)
    want = np.array([[0.0, 0.0], [0.1, 1.0], [0.19, 1.0], [0.271, 1.0]])
    close(path, want, atol=1e-9, label="path")


@check("With step 0.1 the loss falls at every step and the path ends near the bottom (1, 1)")
def test_walk_to_bottom(sub):
    got = has(sub, "gradient_descent")
    path = got(lambda w: A @ w - B, np.array([0.0, 0.0]), 0.1, 100)
    losses = np.array([0.5 * p @ A @ p - B @ p for p in path])
    rises = np.nonzero(np.diff(losses) >= 0)[0]
    if rises.size:
        k = int(rises[0])
        fail(f"the loss did not fall at step {k + 1}: {losses[k]:.6f} then {losses[k + 1]:.6f}")
    close(path[-1], np.array([1.0, 1.0]), atol=1e-3, label="last point of the path")
    equal(bool(losses[-1] < -5.49), True, label="loss close to the minimum -5.5")
