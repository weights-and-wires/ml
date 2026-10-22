"""Checks for the truncation and clipping exercise."""

import numpy as np

from mlcheck import check, close, equal, finite_difference, has, shape


def problem(rng, t_steps=7, n=2, d_x=3, d_h=4):
    W_xh = rng.normal(size=(d_h, d_x)) * 0.5
    W_hh = rng.normal(size=(d_h, d_h)) * 0.5
    b = rng.normal(size=(d_h,)) * 0.1
    h0 = rng.normal(size=(n, d_h)) * 0.3
    x = rng.normal(size=(t_steps, n, d_x)) * 0.6
    return x, h0, W_xh, W_hh, b


@check("chunk_bounds splits 6 steps into pairs and 7 steps into 3, 3, 1")
def test_chunk_bounds(sub):
    bounds = has(sub, "chunk_bounds")
    equal([tuple(p) for p in bounds(6, 2)], [(0, 2), (2, 4), (4, 6)], label="chunk_bounds(6, 2)")
    equal([tuple(p) for p in bounds(7, 3)], [(0, 3), (3, 6), (6, 7)], label="chunk_bounds(7, 3)")


@check("chunk_bounds gives one chunk when k is at least T")
def test_chunk_bounds_long(sub):
    bounds = has(sub, "chunk_bounds")
    equal([tuple(p) for p in bounds(5, 5)], [(0, 5)], label="chunk_bounds(5, 5)")
    equal([tuple(p) for p in bounds(5, 9)], [(0, 5)], label="chunk_bounds(5, 9)")


@check("clip_by_global_norm turns [3] and [4] into [0.6] and [0.8] at a limit of 1")
def test_clip_hand(sub):
    clip = has(sub, "clip_by_global_norm")
    out = clip([np.array([3.0]), np.array([4.0])], 1.0)
    close(out[0], np.array([0.6]), atol=1e-12, label="first array")
    close(out[1], np.array([0.8]), atol=1e-12, label="second array")


@check("clip_by_global_norm leaves a short gradient alone and does not change its input")
def test_clip_short(sub):
    clip = has(sub, "clip_by_global_norm")
    rng = np.random.default_rng(0)
    grads = [rng.normal(size=(3, 4)) * 0.01, rng.normal(size=(4,)) * 0.01]
    before = [g.copy() for g in grads]
    out = clip(grads, 10.0)
    for got, want in zip(out, before):
        close(got, want, atol=1e-14, label="an already-short gradient")
    for g, want in zip(grads, before):
        close(g, want, atol=0.0, label="the input list must not be modified")


@check("clip_by_global_norm scales a long gradient to exactly the limit without turning it")
def test_clip_long(sub):
    clip = has(sub, "clip_by_global_norm")
    rng = np.random.default_rng(1)
    grads = [rng.normal(size=(5, 3)) * 3.0, rng.normal(size=(3,)) * 3.0]
    out = clip(grads, 2.0)
    flat_in = np.concatenate([g.ravel() for g in grads])
    flat_out = np.concatenate([g.ravel() for g in out])
    close(np.linalg.norm(flat_out), 2.0, atol=1e-9, label="global norm after clipping")
    cosine = float(flat_in @ flat_out / (np.linalg.norm(flat_in) * np.linalg.norm(flat_out)))
    close(cosine, 1.0, atol=1e-9, label="cosine between the gradient before and after")
    zero = clip([np.zeros(3), np.zeros((2, 2))], 1.0)
    close(zero[0], np.zeros(3), atol=0.0, label="an all-zero gradient")


@check("truncated_bptt with k equal to T matches full bptt")
def test_truncated_full(sub):
    backward = has(sub, "bptt")
    truncated = has(sub, "truncated_bptt")
    rng = np.random.default_rng(2)
    x, h0, W_xh, W_hh, b = problem(rng)
    dH = rng.normal(size=(x.shape[0],) + h0.shape) * 0.4
    full = backward(dH, x, h0, W_xh, W_hh, b)
    cut = truncated(dH, x, h0, W_xh, W_hh, b, k=x.shape[0])
    for name, got, want in zip(["dx", "dW_xh", "dW_hh", "db"], cut, full[:4]):
        close(got, want, atol=1e-10, label=f"{name} at k = T")


@check("truncated_bptt matches finite differences of a loss whose chunk starting states are constants")
def test_truncated_fd(sub):
    truncated = has(sub, "truncated_bptt")
    forward = has(sub, "rnn_forward")
    bounds = has(sub, "chunk_bounds")
    rng = np.random.default_rng(3)
    x, h0, W_xh, W_hh, b = problem(rng, t_steps=7)
    dH = rng.normal(size=(x.shape[0],) + h0.shape) * 0.4
    k = 3
    dx, dW_xh, dW_hh, db = truncated(dH, x, h0, W_xh, W_hh, b, k)
    shape(dx, x.shape, label="dx")

    h_seq = forward(x, h0, W_xh, W_hh, b)
    starts = {s: (h_seq[s - 1] if s > 0 else h0).copy() for s, _ in bounds(x.shape[0], k)}

    def loss(x_p, Wx, Wh, b_p):
        total = 0.0
        for s, e in bounds(x.shape[0], k):
            total += float(np.sum(dH[s:e] * forward(x_p[s:e], starts[s], Wx, Wh, b_p)))
        return total

    close(dx, finite_difference(lambda v: loss(v, W_xh, W_hh, b), x), rtol=1e-4, atol=1e-6, label="dL/dx")
    close(dW_xh, finite_difference(lambda v: loss(x, v, W_hh, b), W_xh), rtol=1e-4, atol=1e-6, label="dL/dW_xh")
    close(dW_hh, finite_difference(lambda v: loss(x, W_xh, v, b), W_hh), rtol=1e-4, atol=1e-6, label="dL/dW_hh")
    close(db, finite_difference(lambda v: loss(x, W_xh, W_hh, v), b), rtol=1e-4, atol=1e-6, label="dL/db")


@check("A loss on the last step sends no gradient into earlier chunks")
def test_truncated_no_crossing(sub):
    truncated = has(sub, "truncated_bptt")
    W_xh, W_hh, b = np.array([[0.5]]), np.array([[0.8]]), np.zeros(1)
    x = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1, 1)
    dH = np.zeros((6, 1, 1))
    dH[-1] = 1.0
    dx, dW_xh, dW_hh, db = truncated(dH, x, np.zeros((1, 1)), W_xh, W_hh, b, 2)
    close(dx[:4].ravel(), np.zeros(4), atol=1e-14, label="dx for steps 1 to 4 (outside the last chunk)")
    close(dx[4:].ravel(), np.array([0.3603, 0.4805]), atol=1e-4, label="dx for steps 5 and 6")
    close(dW_hh, np.array([[0.4710]]), atol=1e-4, label="dW_hh from the last chunk only")
    close(db, np.array([1.6815]), atol=1e-4, label="db from the last chunk only")
