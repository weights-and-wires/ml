"""Checks for the backpropagation through time exercise."""

import numpy as np

from mlcheck import check, close, finite_difference, has, shape


def problem(rng, t_steps=4, n=2, d_x=3, d_h=4):
    W_xh = rng.normal(size=(d_h, d_x)) * 0.5
    W_hh = rng.normal(size=(d_h, d_h)) * 0.5
    b = rng.normal(size=(d_h,)) * 0.1
    h0 = rng.normal(size=(n, d_h)) * 0.3
    x = rng.normal(size=(t_steps, n, d_x)) * 0.6
    return x, h0, W_xh, W_hh, b


def hand_example():
    W_xh, W_hh, b = np.array([[0.5]]), np.array([[0.8]]), np.zeros(1)
    x = np.array([1.0, 0.5]).reshape(2, 1, 1)
    h0 = np.full((1, 1), 0.3)
    dH = np.array([0.0, 1.0]).reshape(2, 1, 1)
    return x, h0, W_xh, W_hh, b, dH


@check("tanh_backward multiplies by 1 - h^2 and matches a finite difference of tanh")
def test_tanh_backward(sub):
    back = has(sub, "tanh_backward")
    h = np.array([[0.63712314]])
    close(back(np.array([[1.0]]), h), np.array([[0.59407410]]), atol=1e-7, label="slope at h_2")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 4))
    dh = rng.normal(size=(3, 4))
    got = back(dh, np.tanh(z))
    shape(got, z.shape, label="tanh_backward output")
    want = finite_difference(lambda v: float(np.sum(dh * np.tanh(v))), z, eps=1e-6)
    close(got, want, rtol=1e-5, atol=1e-8, label="dz versus finite differences")


@check("step_backward on the second step of the hand example")
def test_step_backward_hand(sub):
    step = has(sub, "step_backward")
    x, h0, W_xh, W_hh, b, dH = hand_example()
    h1, h2 = 0.62914516, 0.63712314
    dx, dh_prev, dW_xh, dW_hh, db = step(
        dH[1], np.array([[h2]]), np.array([[h1]]), x[1], W_xh, W_hh
    )
    close(dx, np.array([[0.29703705]]), atol=1e-6, label="dx_2")
    close(dh_prev, np.array([[0.47525928]]), atol=1e-6, label="dh_1 passed back")
    close(dW_xh, np.array([[0.29703705]]), atol=1e-6, label="dW_xh from step 2")
    close(dW_hh, np.array([[0.37375885]]), atol=1e-6, label="dW_hh from step 2")
    close(db, np.array([0.59407410]), atol=1e-6, label="db from step 2")


@check("step_backward matches finite differences for all five outputs")
def test_step_backward_fd(sub):
    step = has(sub, "step_backward")
    cell = has(sub, "rnn_cell_forward")
    rng = np.random.default_rng(1)
    x, h0, W_xh, W_hh, b = problem(rng, t_steps=1)
    x_t = x[0]
    dh_t = rng.normal(size=h0.shape)
    h_t = cell(x_t, h0, W_xh, W_hh, b)
    dx, dh_prev, dW_xh, dW_hh, db = step(dh_t, h_t, h0, x_t, W_xh, W_hh)
    shape(dx, x_t.shape, label="dx_t")
    shape(dh_prev, h0.shape, label="dh_prev")
    shape(dW_xh, W_xh.shape, label="dW_xh")
    shape(dW_hh, W_hh.shape, label="dW_hh")
    shape(db, b.shape, label="db")

    def loss(x_p, h_p, Wx, Wh, b_p):
        return float(np.sum(dh_t * cell(x_p, h_p, Wx, Wh, b_p)))

    close(dx, finite_difference(lambda v: loss(v, h0, W_xh, W_hh, b), x_t), rtol=1e-5, atol=1e-7, label="dx_t")
    close(dh_prev, finite_difference(lambda v: loss(x_t, v, W_xh, W_hh, b), h0), rtol=1e-5, atol=1e-7, label="dh_prev")
    close(dW_xh, finite_difference(lambda v: loss(x_t, h0, v, W_hh, b), W_xh), rtol=1e-5, atol=1e-7, label="dW_xh")
    close(dW_hh, finite_difference(lambda v: loss(x_t, h0, W_xh, v, b), W_hh), rtol=1e-5, atol=1e-7, label="dW_hh")
    close(db, finite_difference(lambda v: loss(x_t, h0, W_xh, W_hh, v), b), rtol=1e-5, atol=1e-7, label="db")


@check("bptt reproduces the hand example: dW_xh 0.5842, dW_hh 0.4599, db 0.8812, dh0 0.2297")
def test_bptt_hand(sub):
    backward = has(sub, "bptt")
    x, h0, W_xh, W_hh, b, dH = hand_example()
    dx, dW_xh, dW_hh, db, dh0 = backward(dH, x, h0, W_xh, W_hh, b)
    close(dW_xh, np.array([[0.58417748]]), atol=1e-6, label="dW_xh")
    close(dW_hh, np.array([[0.45990097]]), atol=1e-6, label="dW_hh (the h0 term is 0.0861 of it)")
    close(db, np.array([0.88121453]), atol=1e-6, label="db")
    close(dh0, np.array([[0.22971234]]), atol=1e-6, label="dh0")
    close(dx.ravel(), np.array([0.14357021, 0.29703705]), atol=1e-6, label="dx")


@check("bptt matches finite differences for the input, both weights, the bias, and h0")
def test_bptt_fd(sub):
    backward = has(sub, "bptt")
    forward = has(sub, "rnn_forward")
    rng = np.random.default_rng(2)
    x, h0, W_xh, W_hh, b = problem(rng)
    dH = rng.normal(size=(x.shape[0],) + h0.shape) * 0.5
    dx, dW_xh, dW_hh, db, dh0 = backward(dH, x, h0, W_xh, W_hh, b)

    def loss(x_p, h_p, Wx, Wh, b_p):
        return float(np.sum(dH * forward(x_p, h_p, Wx, Wh, b_p)))

    close(dx, finite_difference(lambda v: loss(v, h0, W_xh, W_hh, b), x), rtol=1e-4, atol=1e-6, label="dL/dx")
    close(dh0, finite_difference(lambda v: loss(x, v, W_xh, W_hh, b), h0), rtol=1e-4, atol=1e-6, label="dL/dh0")
    close(dW_xh, finite_difference(lambda v: loss(x, h0, v, W_hh, b), W_xh), rtol=1e-4, atol=1e-6, label="dL/dW_xh")
    close(dW_hh, finite_difference(lambda v: loss(x, h0, W_xh, v, b), W_hh), rtol=1e-4, atol=1e-6, label="dL/dW_hh")
    close(db, finite_difference(lambda v: loss(x, h0, W_xh, W_hh, v), b), rtol=1e-4, atol=1e-6, label="dL/db")


@check("A loss on every step is the sum of the losses on each step alone")
def test_bptt_linear_in_dH(sub):
    backward = has(sub, "bptt")
    rng = np.random.default_rng(3)
    x, h0, W_xh, W_hh, b = problem(rng, t_steps=5)
    dH = rng.normal(size=(5,) + h0.shape)
    whole = backward(dH, x, h0, W_xh, W_hh, b)
    parts = [np.zeros_like(g) for g in whole]
    for t in range(5):
        one = np.zeros_like(dH)
        one[t] = dH[t]
        for acc, g in zip(parts, backward(one, x, h0, W_xh, W_hh, b)):
            acc += g
    names = ["dx", "dW_xh", "dW_hh", "db", "dh0"]
    for name, got, want in zip(names, whole, parts):
        close(got, want, atol=1e-10, label=name)
