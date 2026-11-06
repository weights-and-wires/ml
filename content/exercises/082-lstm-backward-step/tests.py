"""Checks for the LSTM backward step exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_case(seed=0, batch=3, n=4, m=5):
    rng = np.random.default_rng(seed)
    params = {}
    for name in "figo":
        params[f"W{name}"] = 0.4 * rng.normal(size=(n, m))
        params[f"U{name}"] = 0.4 * rng.normal(size=(m, m))
        params[f"b{name}"] = 0.1 * rng.normal(size=m)
    x = 0.5 * rng.normal(size=(batch, n))
    h = 0.5 * rng.normal(size=(batch, m))
    c = 0.5 * rng.normal(size=(batch, m))
    return x, h, c, params


def worked_params():
    params = {}
    for name in "figo":
        params[f"W{name}"] = np.zeros((1, 1))
        params[f"U{name}"] = np.zeros((1, 1))
    params["bf"] = np.array([np.log(0.9 / 0.1)])
    params["bi"] = np.array([np.log(0.2 / 0.8)])
    params["bo"] = np.array([0.0])
    params["bg"] = np.array([np.arctanh(0.6)])
    return params


@check("cell_state_gradient gives 0.6004 on the worked example, and adds both paths")
def test_cell_state_gradient(sub):
    fn = has(sub, "cell_state_gradient")
    t = np.tanh(0.48)
    got = fn(np.array([[1.0]]), np.array([[0.2]]), np.array([[0.5]]), np.array([[t]]))
    shape(got, (1, 1), label="cell_state_gradient output")
    close(got, np.array([[0.2 + 0.5 * (1 - t ** 2)]]), atol=1e-10, label="dc_total")
    close(got, np.array([[0.6004]]), atol=1e-4, label="dc_total, rounded")
    rng = np.random.default_rng(0)
    o = rng.uniform(0.1, 0.9, (3, 4))
    tc = np.tanh(rng.normal(size=(3, 4)))
    zero = np.zeros((3, 4))
    dh, dc = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    close(fn(zero, dc, o, tc), dc, atol=1e-12, label="with dh_new = 0, only the belt path")
    close(fn(dh, zero, o, tc), dh * o * (1 - tc ** 2), atol=1e-12, label="with dc_new = 0, only the hidden path")


@check("lstm_cell_backward returns the right shapes and all twelve parameter gradients")
def test_backward_shapes(sub):
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")
    x, h, c, params = lstm_case()
    _, _, cache = forward(x, h, c, params)
    rng = np.random.default_rng(1)
    dx, dh_prev, dc_prev, dparams = backward(rng.normal(size=h.shape), rng.normal(size=c.shape), cache)
    shape(dx, x.shape, label="dx")
    shape(dh_prev, h.shape, label="dh_prev")
    shape(dc_prev, c.shape, label="dc_prev")
    for name, value in params.items():
        if name not in dparams:
            fail(f"dparams is missing '{name}'")
        shape(dparams[name], value.shape, label=f"dparams['{name}']")


@check("lstm_cell_backward reproduces the worked example: dc_prev = 0.5404, bias gradient 0.0216")
def test_backward_worked_example(sub):
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")
    params = worked_params()
    x, h, c = np.array([[1.0]]), np.zeros((1, 1)), np.array([[0.4]])
    _, _, cache = forward(x, h, c, params)
    dx, dh_prev, dc_prev, dparams = backward(np.array([[1.0]]), np.array([[0.2]]), cache)
    t = np.tanh(0.48)
    dc_total = 0.2 + 0.5 * (1 - t ** 2)
    close(dc_prev, np.array([[dc_total * 0.9]]), atol=1e-9, label="dc_prev")
    close(dc_prev, np.array([[0.5404]]), atol=1e-4, label="dc_prev, rounded")
    close(dparams["bf"], np.array([dc_total * 0.4 * 0.9 * 0.1]), atol=1e-9, label="bf gradient")
    close(dparams["bf"], np.array([0.0216]), atol=1e-4, label="bf gradient, rounded")
    close(dparams["bi"], np.array([dc_total * 0.6 * 0.2 * 0.8]), atol=1e-9, label="bi gradient")
    close(dparams["bo"], np.array([1.0 * t * 0.5 * 0.5]), atol=1e-9, label="bo gradient")


@check("lstm_cell_backward matches finite differences for x, h, and c")
def test_backward_inputs(sub):
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")
    x, h, c, params = lstm_case(seed=3)
    rng = np.random.default_rng(30)
    dh_up = 0.1 * rng.normal(size=h.shape)
    dc_up = 0.1 * rng.normal(size=c.shape)
    _, _, cache = forward(x, h, c, params)
    dx, dh_prev, dc_prev, _ = backward(dh_up, dc_up, cache)

    def loss(xx, hh, cc):
        h_new, c_new, _ = forward(xx, hh, cc, params)
        return float(np.sum(h_new * dh_up) + np.sum(c_new * dc_up))

    close(dx, finite_difference(lambda a: loss(a, h, c), x.copy()), atol=1e-5, rtol=1e-4, label="dx")
    close(dh_prev, finite_difference(lambda a: loss(x, a, c), h.copy()), atol=1e-5, rtol=1e-4, label="dh_prev")
    close(dc_prev, finite_difference(lambda a: loss(x, h, a), c.copy()), atol=1e-5, rtol=1e-4, label="dc_prev")


@check("lstm_cell_backward matches finite differences for every weight and bias")
def test_backward_params(sub):
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")
    x, h, c, params = lstm_case(seed=4)
    rng = np.random.default_rng(40)
    dh_up = 0.1 * rng.normal(size=h.shape)
    dc_up = 0.1 * rng.normal(size=c.shape)
    _, _, cache = forward(x, h, c, params)
    _, _, _, dparams = backward(dh_up, dc_up, cache)

    for name in params:
        def loss(arr, name=name):
            trial = dict(params)
            trial[name] = arr
            h_new, c_new, _ = forward(x, h, c, trial)
            return float(np.sum(h_new * dh_up) + np.sum(c_new * dc_up))

        want = finite_difference(loss, params[name].copy())
        close(dparams[name], want, atol=1e-5, rtol=1e-4, label=f"gradient wrt {name}")


@check("with no gradient from h_new, the belt passes dc_new back times the forget gate")
def test_belt_gradient(sub):
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")
    x, h, c, params = lstm_case(seed=7)
    rng = np.random.default_rng(70)
    dc_up = rng.normal(size=c.shape)
    _, _, cache = forward(x, h, c, params)
    _, _, dc_prev, _ = backward(np.zeros_like(h), dc_up, cache)
    close(dc_prev, dc_up * cache["f"], atol=1e-10, label="dc_prev against dc_new * f")
    params2 = dict(params)
    params2["bf"] = np.full(5, 20.0)
    _, _, cache2 = forward(x, h, c, params2)
    _, _, dc_prev2, _ = backward(np.zeros_like(h), dc_up, cache2)
    close(dc_prev2, dc_up, atol=1e-6, label="dc_prev with the forget gate wide open")
