"""Checks for the vanishing gradients, truncation, and clipping exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


@check("spectral_radius is the largest absolute eigenvalue, also for a rotation and a symmetric matrix")
def test_radius(sub):
    radius = has(sub, "spectral_radius")
    close(radius(np.diag([0.5, 0.9])), 0.9, atol=1e-12, label="rho of diag(0.5, 0.9)")
    close(radius(np.diag([-2.0, 1.0, 0.3])), 2.0, atol=1e-12, label="rho of diag(-2, 1, 0.3), the sign must not matter")
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    close(radius(rotation), 1.0, atol=1e-10, label="rho of a rotation, which has no real eigenvalue")
    close(radius(1.05 * rotation), 1.05, atol=1e-10, label="rho of a scaled rotation")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    S = (A + A.T) / 2
    want = float(np.max(np.abs(np.linalg.eigvalsh(S))))
    close(radius(S), want, atol=1e-10, label="rho of a symmetric matrix")


@check("gradient_scale_over_time computes (kappa * rho) ** T")
def test_scale(sub):
    scale = has(sub, "gradient_scale_over_time")
    close(scale(np.diag([0.5, 0.9]), 1.0, 10), 0.3486784401, rtol=1e-9, label="0.9 ** 10")
    close(scale(np.diag([0.5, 0.9]), 0.5, 3), 0.091125, rtol=1e-9, label="(0.5 * 0.9) ** 3")
    angle = 1.0
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    close(scale(1.05 * rotation, 1.0, 100), 131.50125784630401, rtol=1e-8, label="1.05 ** 100")


@check("steps_until_below returns the smallest T, agrees with counting one step at a time, and gives None when the gradient never falls")
def test_steps(sub):
    steps = has(sub, "steps_until_below")
    if steps(0.9, 1.0, 0.01) != 44:
        fail(f"steps_until_below(0.9, 1.0, 0.01) returned {steps(0.9, 1.0, 0.01)}, expected 44 (0.9**44 = 0.0097, 0.9**43 = 0.0108)")
    if steps(0.5, 1.0, 0.25) != 3:
        fail(f"steps_until_below(0.5, 1.0, 0.25) returned {steps(0.5, 1.0, 0.25)}, expected 3, because 0.5**2 equals 0.25 and is not below it")
    for rho, kappa in [(1.0, 1.0), (1.05, 1.0), (2.0, 0.5)]:
        if steps(rho, kappa, 0.1) is not None:
            fail(f"steps_until_below({rho}, {kappa}, 0.1) should be None because kappa * rho = {rho * kappa} is 1 or more")
    for rho, kappa, threshold in [(0.99, 0.9, 1e-3), (1.2, 0.7, 1e-6), (0.3, 1.0, 0.5), (0.97, 1.0, 1e-4)]:
        rate = rho * kappa
        count = 1
        while rate**count >= threshold:
            count += 1
        got = steps(rho, kappa, threshold)
        if got != count:
            fail(f"steps_until_below({rho}, {kappa}, {threshold}) returned {got}, counting one step at a time gives {count}")


@check("The gradient your bptt measures stays within a factor of two of the prediction at rho = 0.9 and 1.05")
def test_measured_vs_predicted(sub):
    backward = has(sub, "bptt")
    predict = has(sub, "gradient_scale_over_time")
    d, T = 6, 100
    for rho in (0.9, 1.05):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        W_hh = rho * q
        x = np.zeros((T, 1, d))
        dH = np.zeros((T, 1, d))
        g = rng.normal(size=(1, d))
        dH[-1] = g / np.linalg.norm(g)
        h0 = rng.normal(size=(1, d)) * 1e-3
        dh0 = backward(dH, x, h0, np.zeros((d, d)), W_hh, np.zeros(d))[4]
        measured = float(np.linalg.norm(dh0))
        predicted = predict(W_hh, 1.0, T)
        ratio = measured / predicted
        if not 0.5 < ratio < 2.0:
            fail(f"at rho={rho}, measured {measured:.3e} and predicted {predicted:.3e} differ by a factor of {ratio:.2f}")


def problem(rng, t_steps=7, n=2, d_x=3, d_h=4):
    W_xh = rng.normal(size=(d_h, d_x)) * 0.5
    W_hh = rng.normal(size=(d_h, d_h)) * 0.5
    b = rng.normal(size=(d_h,)) * 0.1
    h0 = rng.normal(size=(n, d_h)) * 0.3
    x = rng.normal(size=(t_steps, n, d_x)) * 0.6
    return x, h0, W_xh, W_hh, b


@check("chunk_bounds splits 6 steps into pairs, 7 steps into 3, 3, 1, and gives one chunk when k is at least T")
def test_chunk_bounds(sub):
    bounds = has(sub, "chunk_bounds")
    equal([tuple(p) for p in bounds(6, 2)], [(0, 2), (2, 4), (4, 6)], label="chunk_bounds(6, 2)")
    equal([tuple(p) for p in bounds(7, 3)], [(0, 3), (3, 6), (6, 7)], label="chunk_bounds(7, 3)")
    equal([tuple(p) for p in bounds(5, 5)], [(0, 5)], label="chunk_bounds(5, 5)")
    equal([tuple(p) for p in bounds(5, 9)], [(0, 5)], label="chunk_bounds(5, 9)")


@check("clip_by_global_norm turns [3] and [4] into [0.6] and [0.8] at a limit of 1")
def test_clip_hand(sub):
    clip = has(sub, "clip_by_global_norm")
    out = clip([np.array([3.0]), np.array([4.0])], 1.0)
    close(out[0], np.array([0.6]), atol=1e-12, label="first array")
    close(out[1], np.array([0.8]), atol=1e-12, label="second array")


@check("clip_by_global_norm leaves a short gradient alone, scales a long one to exactly the limit without turning it, and keeps its input")
def test_clip_short_long(sub):
    clip = has(sub, "clip_by_global_norm")
    rng = np.random.default_rng(0)
    grads = [rng.normal(size=(3, 4)) * 0.01, rng.normal(size=(4,)) * 0.01]
    before = [g.copy() for g in grads]
    out = clip(grads, 10.0)
    for got, want in zip(out, before):
        close(got, want, atol=1e-14, label="an already-short gradient")
    for g, want in zip(grads, before):
        close(g, want, atol=0.0, label="the input list must not be modified")
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
