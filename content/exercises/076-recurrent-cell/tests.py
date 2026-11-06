"""Checks for the recurrent cell exercise."""

import numpy as np

from mlcheck import check, close, finite_difference, has, shape


def small_problem(rng, t_steps=5, n=3, d_x=4, d_h=6):
    W_xh = rng.normal(size=(d_h, d_x)) * 0.4
    W_hh = rng.normal(size=(d_h, d_h)) * 0.4
    b = rng.normal(size=(d_h,)) * 0.1
    h0 = rng.normal(size=(n, d_h)) * 0.2
    x = rng.normal(size=(t_steps, n, d_x)) * 0.5
    return x, h0, W_xh, W_hh, b


@check("rnn_cell_forward reproduces the hand example: h_1 = 0.4621 and h_2 = 0.5509")
def test_cell_hand_example(sub):
    cell = has(sub, "rnn_cell_forward")
    W_xh, W_hh, b = np.array([[0.5]]), np.array([[0.8]]), np.zeros(1)
    h1 = cell(np.array([[1.0]]), np.zeros((1, 1)), W_xh, W_hh, b)
    close(h1, np.array([[0.46211716]]), atol=1e-6, label="h_1")
    h2 = cell(np.array([[0.5]]), h1, W_xh, W_hh, b)
    close(h2, np.array([[0.55091475]]), atol=1e-6, label="h_2")


@check("rnn_cell_forward handles a batch with different input and state sizes")
def test_cell_batch(sub):
    cell = has(sub, "rnn_cell_forward")
    rng = np.random.default_rng(0)
    x, h0, W_xh, W_hh, b = small_problem(rng, t_steps=1)
    out = cell(x[0], h0, W_xh, W_hh, b)
    shape(out, h0.shape, label="cell output")
    close(out, np.tanh(x[0] @ W_xh.T + h0 @ W_hh.T + b), atol=1e-10, label="cell output")


@check("rnn_forward returns h_1 .. h_T and matches the six-step trace of the running example")
def test_forward_trace(sub):
    forward = has(sub, "rnn_forward")
    W_xh, W_hh, b = np.array([[0.5]]), np.array([[0.8]]), np.zeros(1)
    x = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1, 1)
    h_seq = forward(x, np.zeros((1, 1)), W_xh, W_hh, b)
    shape(h_seq, (6, 1, 1), label="rnn_forward output")
    want = np.array([0.4621, 0.5509, 0.4143, 0.3198, 0.2504, 0.1977])
    close(h_seq[:, 0, 0], want, atol=1e-4, label="the six states")


@check("rnn_forward starts from h0, and reading a prefix gives the first rows of the full run")
def test_forward_prefix(sub):
    forward = has(sub, "rnn_forward")
    cell = has(sub, "rnn_cell_forward")
    rng = np.random.default_rng(1)
    x, h0, W_xh, W_hh, b = small_problem(rng, t_steps=6)
    full = forward(x, h0, W_xh, W_hh, b)
    shape(full, (6,) + h0.shape, label="rnn_forward output")
    close(full[0], cell(x[0], h0, W_xh, W_hh, b), atol=1e-10, label="row 0 (h_1 from h0)")
    prefix = forward(x[:3], h0, W_xh, W_hh, b)
    close(prefix, full[:3], atol=1e-10, label="a 3-step run versus the first 3 rows")


@check("last_step_loss_and_dH matches the hand example: loss 0.005185, dH 0.2037 at the last step")
def test_loss_hand_example(sub):
    loss_fn = has(sub, "last_step_loss_and_dH")
    h_seq = np.array([0.46211716, 0.55091475]).reshape(2, 1, 1)
    loss, dH = loss_fn(h_seq, np.array([[2.0]]), np.zeros(1), np.array([[1.0]]))
    close(loss, 0.0051846, atol=1e-6, label="loss")
    shape(dH, (2, 1, 1), label="dH")
    close(dH[:, 0, 0], np.array([0.0, 0.20365899]), atol=1e-6, label="dH")


@check("last_step_loss_and_dH puts gradient only on the last step, and it matches finite differences")
def test_loss_finite_difference(sub):
    loss_fn = has(sub, "last_step_loss_and_dH")
    rng = np.random.default_rng(2)
    t_steps, n, d_h, d_o = 4, 3, 5, 2
    h_seq = rng.normal(size=(t_steps, n, d_h)) * 0.5
    W_hy = rng.normal(size=(d_o, d_h))
    b_y = rng.normal(size=(d_o,))
    y = rng.normal(size=(n, d_o))
    _, dH = loss_fn(h_seq, W_hy, b_y, y)
    shape(dH, h_seq.shape, label="dH")
    if np.abs(dH[:-1]).max() != 0.0:
        raise AssertionError("dH must be zero at every step except the last")
    want = finite_difference(lambda v: loss_fn(v, W_hy, b_y, y)[0], h_seq, eps=1e-6)
    close(dH, want, rtol=1e-5, atol=1e-7, label="dH versus finite differences")
