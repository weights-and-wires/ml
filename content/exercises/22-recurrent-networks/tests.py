"""Checks for the recurrent-networks exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def small_problem(rng, t_steps=5, n=3, d_x=4, d_h=6):
    W_xh = rng.normal(size=(d_h, d_x)) * 0.4
    W_hh = rng.normal(size=(d_h, d_h)) * 0.4
    b = rng.normal(size=(d_h,)) * 0.1
    h0 = rng.normal(size=(n, d_h)) * 0.2
    x = rng.normal(size=(t_steps, n, d_x)) * 0.5
    return x, h0, W_xh, W_hh, b


@check("rnn_cell_forward matches tanh(x W_xh^T + h W_hh^T + b) directly")
def test_rnn_cell_forward(sub):
    got = has(sub, "rnn_cell_forward")
    rng = np.random.default_rng(0)
    x_t, h0, W_xh, W_hh, b = small_problem(rng, t_steps=1)
    h_out = got(x_t[0], h0, W_xh, W_hh, b)
    shape(h_out, h0.shape, label="rnn_cell_forward output")
    want = np.tanh(x_t[0] @ W_xh.T + h0 @ W_hh.T + b)
    close(h_out, want, atol=1e-10, label="cell output")


@check("rnn_forward matches repeated application of rnn_cell_forward")
def test_rnn_forward(sub):
    cell = has(sub, "rnn_cell_forward")
    got = has(sub, "rnn_forward")
    rng = np.random.default_rng(1)
    x, h0, W_xh, W_hh, b = small_problem(rng)

    h_seq = got(x, h0, W_xh, W_hh, b)
    shape(h_seq, (x.shape[0],) + h0.shape, label="rnn_forward output")

    h_prev = h0
    want = np.empty_like(h_seq)
    for t in range(x.shape[0]):
        h_prev = cell(x[t], h_prev, W_xh, W_hh, b)
        want[t] = h_prev
    close(h_seq, want, atol=1e-8, label="hidden sequence")


@check("bptt matches finite differences for the input, both weights, the bias, and h0")
def test_bptt_finite_difference(sub):
    forward = has(sub, "rnn_forward")
    backward = has(sub, "bptt")
    rng = np.random.default_rng(2)
    x, h0, W_xh, W_hh, b = small_problem(rng, t_steps=4, n=2, d_x=3, d_h=4)
    dH = rng.normal(size=(x.shape[0], h0.shape[0], h0.shape[1])) * 0.3

    dx, dW_xh, dW_hh, db, dh0 = backward(dH, x, h0, W_xh, W_hh, b)
    shape(dx, x.shape, label="dx")
    shape(dW_xh, W_xh.shape, label="dW_xh")
    shape(dW_hh, W_hh.shape, label="dW_hh")
    shape(db, b.shape, label="db")
    shape(dh0, h0.shape, label="dh0")

    def loss(x_p, h0_p, W_xh_p, W_hh_p, b_p):
        h_seq = forward(x_p, h0_p, W_xh_p, W_hh_p, b_p)
        return float(np.sum(dH * h_seq))

    close(dx, finite_difference(lambda v: loss(v, h0, W_xh, W_hh, b), x, eps=1e-5),
          rtol=1e-3, atol=1e-4, label="dL/dx")
    close(dh0, finite_difference(lambda v: loss(x, v, W_xh, W_hh, b), h0, eps=1e-5),
          rtol=1e-3, atol=1e-4, label="dL/dh0")
    close(dW_xh, finite_difference(lambda v: loss(x, h0, v, W_hh, b), W_xh, eps=1e-5),
          rtol=1e-3, atol=1e-4, label="dL/dW_xh")
    close(dW_hh, finite_difference(lambda v: loss(x, h0, W_xh, v, b), W_hh, eps=1e-5),
          rtol=1e-3, atol=1e-4, label="dL/dW_hh")
    close(db, finite_difference(lambda v: loss(x, h0, W_xh, W_hh, v), b, eps=1e-5),
          rtol=1e-3, atol=1e-4, label="dL/db")


@check("truncated_bptt with k equal to the sequence length matches full bptt exactly")
def test_truncated_matches_full(sub):
    backward = has(sub, "bptt")
    truncated = has(sub, "truncated_bptt")
    rng = np.random.default_rng(3)
    x, h0, W_xh, W_hh, b = small_problem(rng, t_steps=7, n=2, d_x=3, d_h=5)
    dH = rng.normal(size=(x.shape[0], h0.shape[0], h0.shape[1])) * 0.2

    dx_full, dW_xh_full, dW_hh_full, db_full, _ = backward(dH, x, h0, W_xh, W_hh, b)
    dx_t, dW_xh_t, dW_hh_t, db_t = truncated(dH, x, h0, W_xh, W_hh, b, k=x.shape[0])

    close(dx_t, dx_full, atol=1e-10, label="dx at k = T")
    close(dW_xh_t, dW_xh_full, atol=1e-10, label="dW_xh at k = T")
    close(dW_hh_t, dW_hh_full, atol=1e-10, label="dW_hh at k = T")
    close(db_t, db_full, atol=1e-10, label="db at k = T")


@check("truncated_bptt with a short window disagrees with full bptt")
def test_truncated_differs_when_short(sub):
    backward = has(sub, "bptt")
    truncated = has(sub, "truncated_bptt")
    rng = np.random.default_rng(4)
    x, h0, W_xh, W_hh, b = small_problem(rng, t_steps=8, n=2, d_x=3, d_h=5)
    dH = rng.normal(size=(x.shape[0], h0.shape[0], h0.shape[1])) * 0.2

    _, _, dW_hh_full, _, _ = backward(dH, x, h0, W_xh, W_hh, b)
    _, _, dW_hh_trunc, _ = truncated(dH, x, h0, W_xh, W_hh, b, k=2)

    if np.allclose(dW_hh_full, dW_hh_trunc, atol=1e-8):
        fail(
            "truncating to a window of 2 steps on an 8-step sequence should drop "
            "cross-boundary gradient terms, but the weight gradient is identical "
            "to full bptt"
        )


@check("clip_by_global_norm leaves an under-threshold gradient unchanged")
def test_clip_under_threshold(sub):
    got = has(sub, "clip_by_global_norm")
    rng = np.random.default_rng(5)
    grads = [rng.normal(size=(3, 4)) * 0.01, rng.normal(size=(4,)) * 0.01]
    clipped = got(grads, max_norm=10.0)
    for g, c in zip(grads, clipped):
        close(c, g, atol=1e-12, label="an already-small gradient")


@check("clip_by_global_norm rescales an over-threshold gradient to exactly the threshold, preserving direction")
def test_clip_over_threshold(sub):
    got = has(sub, "clip_by_global_norm")
    rng = np.random.default_rng(6)
    grads = [rng.normal(size=(5, 3)) * 3.0, rng.normal(size=(3,)) * 3.0]
    max_norm = 1.0
    clipped = got(grads, max_norm)

    global_norm_after = float(np.sqrt(sum(np.sum(np.asarray(c) ** 2) for c in clipped)))
    if abs(global_norm_after - max_norm) > 1e-6:
        fail(f"clipped global norm is {global_norm_after:.6g}, expected exactly {max_norm}")

    flat_before = np.concatenate([g.ravel() for g in grads])
    flat_after = np.concatenate([c.ravel() for c in clipped])
    cosine = float(flat_before @ flat_after / (np.linalg.norm(flat_before) * np.linalg.norm(flat_after)))
    if cosine < 1.0 - 1e-8:
        fail(f"clipping changed the gradient's direction: cosine similarity is {cosine:.10f}, expected 1.0")


@check("gradient_scale_over_time matches an independent spectral-radius computation")
def test_gradient_scale_formula(sub):
    got = has(sub, "gradient_scale_over_time")
    rng = np.random.default_rng(7)
    W = rng.normal(size=(6, 6)) * 0.3
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    for kappa, T in [(1.0, 20), (0.5, 10)]:
        want = (kappa * rho) ** T
        got_value = got(W, kappa, T)
        rel = abs(got_value - want) / max(abs(want), 1e-300)
        if rel > 1e-6:
            fail(
                f"gradient_scale_over_time(W, {kappa}, {T}) returned {got_value:.6g}, "
                f"expected {want:.6g} from (kappa * rho(W))**T"
            )


@check("The measured gradient magnitude over 100 steps tracks the spectral-radius prediction")
def test_measured_gradient_tracks_prediction(sub):
    backward = has(sub, "bptt")
    predict = has(sub, "gradient_scale_over_time")

    d, n, t_steps = 6, 1, 100
    for rho in (0.9, 1.05):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        W_hh = rho * q
        W_xh = np.zeros((d, d))
        b = np.zeros(d)
        h0 = rng.normal(size=(n, d)) * 1e-3
        x = np.zeros((t_steps, n, d))
        dH = np.zeros((t_steps, n, d))
        g = rng.normal(size=(n, d))
        dH[-1] = g / np.linalg.norm(g)

        _, _, _, _, dh0 = backward(dH, x, h0, W_xh, W_hh, b)
        measured = float(np.linalg.norm(dh0))
        predicted = float(predict(W_hh, 1.0, t_steps))

        if predicted <= 0 or not np.isfinite(predicted):
            fail(f"gradient_scale_over_time returned a non-positive or non-finite value: {predicted}")

        log_ratio = np.log10(max(measured, 1e-300) / predicted)
        if abs(log_ratio) > 1.0:
            fail(
                f"at rho={rho}, measured gradient norm {measured:.3e} should be within an "
                f"order of magnitude of the predicted {predicted:.3e}, but the ratio is "
                f"10**{log_ratio:.2f}"
            )
