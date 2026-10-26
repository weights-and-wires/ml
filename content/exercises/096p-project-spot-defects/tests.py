"""Checks for the defect-spotting project."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


def naive_conv(x, w, b, stride=1, padding=0):
    n, c, h, wd = x.shape
    c_out, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (wd + 2 * padding - k) // stride + 1
    out = np.zeros((n, c_out, h_out, w_out))
    for i in range(n):
        for o in range(c_out):
            for r in range(h_out):
                for q in range(w_out):
                    patch = xp[i, :, r * stride : r * stride + k, q * stride : q * stride + k]
                    out[i, o, r, q] = b[o] + np.sum(w[o] * patch)
    return out


_cache = {}


def trained(sub):
    """Train the CNN and the MLP once per submission and share the result."""
    key = id(sub)
    if key not in _cache:
        X_tr, y_tr, X_te, y_te = sub.make_data(seed=0)
        cnn, _ = sub.fit(sub.cnn_forward, sub.cnn_backward, sub.init_cnn(0), X_tr, y_tr, epochs=20)
        mlp, _ = sub.fit(sub.mlp_forward, sub.mlp_backward, sub.init_mlp(0), X_tr, y_tr, epochs=20)
        _cache[key] = (cnn, mlp, X_te, y_te)
    return _cache[key]


@check("receptive_field counts what a unit sees, including strides")
def test_receptive_field(sub):
    rf = has(sub, "receptive_field")
    equal(rf([(3, 1)]), 3, label="one 3x3 layer")
    equal(rf([(3, 1), (3, 1)]), 5, label="two 3x3 layers")
    equal(rf([(3, 2), (3, 1)]), 7, label="stride 2 then stride 1")
    equal(rf([(2, 2), (2, 2)]), 4, label="two 2x2 layers of stride 2")
    got = rf([(3, 1), (3, 1), (3, 1)])
    if got < sub.SCRATCH_LEN:
        fail(f"the stem plus the residual block should see {sub.SCRATCH_LEN} pixels or more, got {got}")


@check("im2col lays windows out as columns in (channel, row, column) order")
def test_im2col(sub):
    x = np.arange(2 * 2 * 4 * 4, dtype=float).reshape(2, 2, 4, 4)
    col = has(sub, "im2col")(x, 2, 1, 0)
    shape(col, (2, 8, 9), label="im2col shape for k=2")
    # Column 4 is output position (1, 1): rows 1..2, columns 1..2 of each channel.
    want = x[0, :, 1:3, 1:3].reshape(-1)
    close(col[0, :, 4], want, label="column for position (1, 1)")
    col2 = sub.im2col(x, 3, 2, 1)
    shape(col2, (2, 18, 4), label="im2col shape for k=3, stride 2, padding 1")
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    close(col2[1, :, 3], xp[1, :, 2:5, 2:5].reshape(-1), label="last column, stride 2 padding 1")


@check("conv_forward matches a plain loop for several strides and paddings")
def test_conv_forward(sub):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 2, 7, 6))
    w = rng.normal(size=(4, 2, 3, 3))
    b = rng.normal(size=4)
    for stride, padding in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        got = has(sub, "conv_forward")(x, w, b, stride, padding)
        close(got, naive_conv(x, w, b, stride, padding), atol=1e-9, label=f"stride {stride}, padding {padding}")


@check("conv_backward matches finite differences for dx, dw, and db")
def test_conv_backward(sub):
    rng = np.random.default_rng(2)
    for stride, padding in [(1, 1), (2, 1)]:
        x = rng.normal(size=(2, 2, 5, 5))
        w = rng.normal(size=(3, 2, 3, 3))
        b = rng.normal(size=3)
        out = sub.conv_forward(x, w, b, stride, padding)
        dY = rng.normal(size=out.shape)
        dx, dw, db = has(sub, "conv_backward")(dY, x, w, stride, padding)
        shape(dx, x.shape, label="dx")
        shape(dw, w.shape, label="dw")
        shape(db, b.shape, label="db")
        loss = lambda xx, ww, bb: float(np.sum(dY * sub.conv_forward(xx, ww, bb, stride, padding)))
        tag = f"stride {stride}"
        close(dx, finite_difference(lambda v: loss(v, w, b), x), atol=1e-5, label=f"dx, {tag}")
        close(dw, finite_difference(lambda v: loss(x, v, b), w), atol=1e-5, label=f"dw, {tag}")
        close(db, finite_difference(lambda v: loss(x, w, v), b), atol=1e-5, label=f"db, {tag}")


def _block_params(rng, c, scale=0.4):
    return {
        "W1": rng.normal(size=(c, c, 3, 3)) * scale, "b1": rng.normal(size=c) * 0.1 + 0.05,
        "W2": rng.normal(size=(c, c, 3, 3)) * scale, "b2": rng.normal(size=c) * 0.1,
    }


@check("res_block_forward adds the input to the two-convolution branch")
def test_res_forward(sub):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 3, 5, 5))
    p = _block_params(rng, 3)
    y, _ = has(sub, "res_block_forward")(x, p)
    h1 = naive_conv(x, p["W1"], p["b1"], 1, 1)
    want = x + naive_conv(np.maximum(h1, 0), p["W2"], p["b2"], 1, 1)
    close(y, want, atol=1e-9, label="block output")
    zero = {k: np.zeros_like(v) for k, v in p.items()}
    y0, _ = sub.res_block_forward(x, zero)
    close(y0, x, label="block with all-zero weights must be the identity")


@check("res_block_backward matches finite differences for the input and every weight")
def test_res_backward(sub):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 2, 5, 5))
    p = _block_params(rng, 2)
    y, cache = sub.res_block_forward(x, p)
    dY = rng.normal(size=y.shape)
    dx, grads = has(sub, "res_block_backward")(dY, cache)

    def loss_x(v):
        return float(np.sum(dY * sub.res_block_forward(v, p)[0]))

    close(dx, finite_difference(loss_x, x), atol=1e-5, label="dx (shortcut plus branch)")
    for key in ["W1", "b1", "W2", "b2"]:
        def loss_p(v, key=key):
            q = dict(p)
            q[key] = v
            return float(np.sum(dY * sub.res_block_forward(x, q)[0]))

        shape(grads[key], p[key].shape, label=f"d{key}")
        close(grads[key], finite_difference(loss_p, p[key]), atol=1e-5, label=f"d{key}")


@check("Global average pooling averages each channel and spreads its gradient evenly")
def test_gap(sub):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 4, 5, 6))
    out = has(sub, "gap_forward")(x)
    shape(out, (3, 4), label="pooled shape")
    close(out, x.reshape(3, 4, -1).mean(axis=2), label="pooled values")
    dfeat = rng.normal(size=(3, 4))
    dx = has(sub, "gap_backward")(dfeat, x.shape)
    close(dx, finite_difference(lambda v: float(np.sum(dfeat * sub.gap_forward(v))), x), atol=1e-6, label="gap gradient")


@check("The whole CNN's gradients match finite differences for every parameter")
def test_cnn_gradients(sub):
    rng = np.random.default_rng(6)
    params = sub.init_cnn(seed=1, channels=3)
    for key in ["b0", "b1", "b2"]:
        params[key] = rng.normal(size=3) * 0.1 + 0.1
    params["bo"] = np.array([0.1])
    X = rng.normal(size=(4, 1, 6, 6))
    y = np.array([0.0, 1.0, 1.0, 0.0])

    def loss_fn(p):
        logits, _ = has(sub, "cnn_forward")(p, X)
        return sub.bce_with_logits(logits, y)[0]

    logits, cache = sub.cnn_forward(params, X)
    shape(logits, (4,), label="logits")
    _, dlogits = sub.bce_with_logits(logits, y)
    grads = has(sub, "cnn_backward")(params, cache, dlogits)
    for key in params:
        if key not in grads:
            fail(f"cnn_backward returned no gradient for '{key}'; keys: {sorted(grads)}")

        def f(v, key=key):
            q = dict(params)
            q[key] = v
            return loss_fn(q)

        close(grads[key], finite_difference(f, params[key]), atol=1e-6, rtol=1e-4, label=f"gradient of {key}")


@check("adam_step follows the bias-corrected Adam update and minimises a bowl")
def test_adam(sub):
    adam = has(sub, "adam_step")
    p = np.array([1.0, -2.0])
    g = np.array([0.5, -0.25])
    state = {"m": np.zeros(2), "v": np.zeros(2), "t": 0}
    p1 = adam(p, g, state, lr=0.1)
    equal(state["t"], 1, label="step counter after one step")
    close(p1, p - 0.1 * g / (np.abs(g) + 1e-8), atol=1e-9, label="first step moves by about lr in the gradient's sign")
    g2 = np.array([0.1, 0.3])
    p2 = adam(p1, g2, state, lr=0.1)
    m = 0.9 * (0.1 * g) + 0.1 * g2
    v = 0.999 * (0.001 * g**2) + 0.001 * g2**2
    want = p1 - 0.1 * (m / (1 - 0.9**2)) / (np.sqrt(v / (1 - 0.999**2)) + 1e-8)
    close(p2, want, atol=1e-9, label="second step")
    target = np.array([3.0, -1.0])
    x = np.zeros(2)
    st = {"m": np.zeros(2), "v": np.zeros(2), "t": 0}
    for _ in range(400):
        x = adam(x, 2 * (x - target), st, lr=0.05)
    close(x, target, atol=0.05, label="minimiser of a quadratic bowl")


@check("The trained CNN reaches 90% test accuracy on unseen pictures")
def test_cnn_accuracy(sub):
    cnn, _, X_te, y_te = trained(sub)
    acc = sub.accuracy(sub.cnn_forward, cnn, X_te, y_te)
    if acc < 0.90:
        fail(f"CNN test accuracy {acc:.3f}, expected at least 0.90")


@check("The CNN beats an MLP with a similar parameter count by 20 points")
def test_cnn_beats_mlp(sub):
    cnn, mlp, X_te, y_te = trained(sub)
    n_cnn, n_mlp = sub.count_params(cnn), sub.count_params(mlp)
    if abs(n_cnn - n_mlp) > 0.25 * n_cnn:
        fail(f"parameter counts differ too much to compare fairly: CNN {n_cnn}, MLP {n_mlp}")
    a_cnn = sub.accuracy(sub.cnn_forward, cnn, X_te, y_te)
    a_mlp = sub.accuracy(sub.mlp_forward, mlp, X_te, y_te)
    if a_cnn - a_mlp < 0.20:
        fail(f"expected the CNN to lead by 0.20, got CNN {a_cnn:.3f} against MLP {a_mlp:.3f}")


@check("Moving a defect does not change the CNN's prediction much")
def test_translation(sub):
    cnn, _, X_te, y_te = trained(sub)
    base = sub.accuracy(sub.cnn_forward, cnn, X_te, y_te)
    p = lambda X: 1.0 / (1.0 + np.exp(-sub.cnn_forward(cnn, X)[0]))
    for shift in [(2, -3), (-3, 1)]:
        moved = np.roll(X_te, shift=shift, axis=(2, 3))
        acc = sub.accuracy(sub.cnn_forward, cnn, moved, y_te)
        if acc < base - 0.12 or acc < 0.80:
            fail(f"after shifting by {shift}, accuracy fell from {base:.3f} to {acc:.3f}")
        change = float(np.mean(np.abs(p(X_te) - p(moved))))
        if change > 0.25:
            fail(f"after shifting by {shift}, the mean change in probability was {change:.3f}, expected under 0.25")
