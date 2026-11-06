"""Checks for the gradient-through-deep-stacks exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


@check("orthogonal_matrix returns a matrix with Q.T @ Q equal to the identity")
def test_orthogonal(sub):
    orthogonal_matrix = has(sub, "orthogonal_matrix")
    rng = np.random.default_rng(0)
    for d in (2, 5, 16):
        q = orthogonal_matrix(d, rng)
        shape(q, (d, d), label=f"orthogonal_matrix({d}) output")
        close(q.T @ q, np.eye(d), atol=1e-10, label=f"Q.T @ Q for d = {d}")


@check("orthogonal_matrix keeps a vector's length and gives different matrices on different draws")
def test_orthogonal_length(sub):
    orthogonal_matrix = has(sub, "orthogonal_matrix")
    rng = np.random.default_rng(1)
    q1 = orthogonal_matrix(8, rng)
    q2 = orthogonal_matrix(8, rng)
    v = rng.normal(size=8)
    almost(np.linalg.norm(v @ q1), np.linalg.norm(v), tol=1e-10, label="length of v @ Q")
    if np.allclose(q1, q2):
        fail("two draws from the same generator gave the same matrix; draw from rng each time")


@check("scalar_gradient_factor is J ** depth for a plain stack and (1 + J) ** depth for a residual stack")
def test_scalar(sub):
    factor = has(sub, "scalar_gradient_factor")
    almost(factor(0.1, 50, False), 0.1**50, tol=1e-60, label="plain, J = 0.1, 50 layers")
    almost(factor(0.1, 3, True), 1.1**3, tol=1e-12, label="residual, J = 0.1, 3 layers")
    almost(factor(0.1, 50, True), 1.1**50, tol=1e-6, label="residual, J = 0.1, 50 layers")
    almost(factor(0.5, 0, True), 1.0, tol=1e-12, label="zero layers change nothing")


@check("The plain stack shortens the gradient by exactly 0.1 per layer")
def test_plain_exact(sub):
    got = has(sub, "gradient_norm_through_stack")
    for depth in (1, 10, 50):
        value = got(depth, False)
        want = 0.1**depth
        if not abs(value - want) <= 1e-6 * want:
            fail(
                f"a plain stack of {depth} layers, each 0.1 times an orthogonal matrix, "
                f"should leave a gradient of length 0.1**{depth} = {want:.3e}, but got {value:.3e}. "
                "If the number is close but not equal, the matrices are not orthogonal."
            )


@check("The residual stack keeps the gradient within an order of magnitude of one at depth 50")
def test_residual_alive(sub):
    got = has(sub, "gradient_norm_through_stack")
    value = got(50, True)
    if not (0.1 <= value <= 10.0):
        fail(f"a 50-layer residual stack should keep the gradient length between 0.1 and 10, but got {value:.3g}")
    one = got(1, True)
    if not (0.9 <= one <= 1.1):
        fail(f"one residual layer of scale 0.1 changes the length by at most 10 percent, but the length went from 1 to {one:.3g}")


@check("The residual length is not what the scalar model predicts, and a zero-depth stack returns 1")
def test_scalar_overshoots(sub):
    got = has(sub, "gradient_norm_through_stack")
    factor = has(sub, "scalar_gradient_factor")
    almost(got(0, True), 1.0, tol=1e-12, label="zero-depth residual stack")
    almost(got(0, False), 1.0, tol=1e-12, label="zero-depth plain stack")
    toy = factor(0.1, 50, True)
    value = got(50, True)
    if not value < toy / 10:
        fail(
            f"the scalar model predicts growth to {toy:.3g}, but notes that point at angles "
            f"should keep the real length far below that; your stack gave {value:.3g}"
        )


@check("The gap between the residual and the plain stack widens with depth")
def test_gap(sub):
    got = has(sub, "gradient_norm_through_stack")
    shallow = got(10, True) / max(got(10, False), 1e-300)
    deep = got(50, True) / max(got(50, False), 1e-300)
    if not deep > shallow * 1e10:
        fail(f"the residual-to-plain ratio should grow by many orders of magnitude from depth 10 to 50, but went from {shallow:.3g} to {deep:.3g}")
