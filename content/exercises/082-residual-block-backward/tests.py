"""Checks for the residual-block backward exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def worked_example():
    params = {
        "W1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b1": np.array([0.0, -1.0]),
        "W2": np.array([[2.0, 0.0], [0.0, 0.5]]),
        "b2": np.array([0.1, 0.0]),
    }
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    return x, params


def random_case(seed, n=3, d=4):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(size=(d, d)) * 0.5,
        "b1": rng.normal(size=(d,)) * 0.1,
        "W2": rng.normal(size=(d, d)) * 0.5,
        "b2": rng.normal(size=(d,)) * 0.1,
    }
    x = rng.normal(size=(n, d)) * 0.5
    dY = rng.normal(size=(n, d))
    return x, params, dY


@check("relu_mask is 1.0 where the input is positive and 0.0 elsewhere, including at 0")
def test_relu_mask(sub):
    relu_mask = has(sub, "relu_mask")
    h = np.array([[1.0, -1.0, 0.0], [2.5, -0.001, 1e-9]])
    out = relu_mask(h)
    shape(out, (2, 3), label="relu_mask output")
    close(out, np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]), atol=0.0, label="mask values")
    if np.asarray(out).dtype.kind != "f":
        fail(f"relu_mask should return floats, got dtype {np.asarray(out).dtype}")


@check("branch_gradient returns the branch's share of dx on the worked example")
def test_branch_worked(sub):
    forward = has(sub, "residual_block_forward")
    branch_gradient = has(sub, "branch_gradient")
    x, params = worked_example()
    _, cache = forward(x, params)
    got = branch_gradient(np.ones((2, 2)), cache)
    shape(got, (2, 2), label="branch_gradient output")
    close(got, np.array([[2.0, 0.5], [0.0, 0.5]]), atol=1e-10, label="branch term")


@check("branch_gradient equals dx minus dY from finite differences, and is zero when W1 is zero")
def test_branch_fd(sub):
    forward = has(sub, "residual_block_forward")
    branch_gradient = has(sub, "branch_gradient")
    x, params, dY = random_case(5)
    _, cache = forward(x, params)

    def loss_x(x_probe):
        y_probe, _ = forward(x_probe, params)
        return float(np.sum(dY * y_probe))

    numeric_dx = finite_difference(loss_x, x, eps=1e-5)
    close(branch_gradient(dY, cache), numeric_dx - dY, rtol=1e-3, atol=1e-4, label="branch term vs finite differences")

    flat = dict(params)
    flat["W1"] = np.zeros_like(params["W1"])
    _, cache0 = forward(x, flat)
    close(branch_gradient(dY, cache0), np.zeros_like(dY), atol=1e-12, label="branch term with W1 = 0")


@check("residual_block_backward reproduces every gradient in the worked example")
def test_backward_worked(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    x, params = worked_example()
    _, cache = forward(x, params)
    dx, grads = backward(np.ones((2, 2)), cache)
    shape(dx, (2, 2), label="dx")
    close(dx, np.array([[3.0, 1.5], [1.0, 1.5]]), atol=1e-10, label="dx")
    close(grads["W2"], np.array([[1.0, 3.0], [1.0, 3.0]]), atol=1e-10, label="dW2")
    close(grads["b2"], np.array([2.0, 2.0]), atol=1e-10, label="db2")
    close(grads["W1"], np.array([[2.0, 4.0], [0.0, 2.5]]), atol=1e-10, label="dW1")
    close(grads["b1"], np.array([2.0, 1.0]), atol=1e-10, label="db1")


@check("residual_block_backward matches finite differences for x and every parameter")
def test_backward_fd(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    x, params, dY = random_case(2)
    y, cache = forward(x, params)
    dx, grads = backward(dY, cache)
    shape(dx, x.shape, label="dx")

    def loss_x(x_probe):
        y_probe, _ = forward(x_probe, params)
        return float(np.sum(dY * y_probe))

    close(dx, finite_difference(loss_x, x, eps=1e-5), rtol=1e-3, atol=1e-4, label="dL/dx")

    for key in ("W1", "b1", "W2", "b2"):

        def loss_param(value, key=key):
            trial = dict(params)
            trial[key] = value
            y_probe, _ = forward(x, trial)
            return float(np.sum(dY * y_probe))

        shape(grads[key], params[key].shape, label=f"grad for {key}")
        numeric = finite_difference(loss_param, params[key], eps=1e-5)
        close(grads[key], numeric, rtol=1e-3, atol=1e-4, label=f"dL/d{key}")


@check("With W1 zero the branch sends nothing back, so dx equals dY exactly")
def test_shortcut_share(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    x, params, dY = random_case(7)
    params["W1"] = np.zeros_like(params["W1"])
    _, cache = forward(x, params)
    dx, _ = backward(dY, cache)
    close(dx, dY, atol=1e-12, label="dx with W1 = 0 (the shortcut's share alone)")


@check("residual_block_backward leaves dY alone and works on a wider block")
def test_backward_inputs_untouched(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    x, params, dY = random_case(11, n=6, d=5)
    dY_copy = dY.copy()
    _, cache = forward(x, params)
    dx, grads = backward(dY, cache)
    close(dY, dY_copy, atol=0.0, label="dY after the call (it must not change)")
    shape(dx, (6, 5), label="dx")
    for key in grads:
        shape(grads[key], params[key].shape, label=f"grad for {key}")
