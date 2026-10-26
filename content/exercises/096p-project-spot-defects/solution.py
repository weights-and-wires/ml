"""Reference solution: spot defects on a production line with a small residual CNN."""

import numpy as np

SIZE = 14  # side length of every picture
SCRATCH_LEN = 7  # a scratch spans at most this many pixels along either axis


# ---------------------------------------------------------------------------
# Data (solved for you)
# ---------------------------------------------------------------------------
def make_data(seed: int = 0, n_train: int = 1000, n_test: int = 400):
    """Make greyscale pictures of parts, half of them defective.

    Every picture is SIZE by SIZE. The background is a smooth stripe texture
    with a random direction, phase, and amplitude, plus Gaussian pixel noise.
    A defective picture (label 1) also holds one defect at a random place:
    a bright scratch, a line of SCRATCH_LEN pixels at a random angle, or a dark
    dent, a small Gaussian blob. A good picture (label 0) holds neither.

    Returns:
        (X_train, y_train, X_test, y_test). Images have shape (N, 1, SIZE,
        SIZE) and dtype float64. Labels are float arrays of shape (N,) holding
        0.0 or 1.0, exactly half ones.
    """
    rng = np.random.default_rng(seed)

    def build(n):
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
        X = np.empty((n, 1, SIZE, SIZE))
        y = np.zeros(n)
        y[: n // 2] = 1.0
        rng.shuffle(y)
        for i in range(n):
            theta = rng.uniform(0, np.pi)
            freq = rng.uniform(0.25, 0.7)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.2, 0.5)
            u = xx * np.cos(theta) + yy * np.sin(theta)
            img = amp * np.sin(freq * u + phase)
            if y[i] == 1.0:
                cy, cx = rng.uniform(2.5, SIZE - 3.5, size=2)
                if rng.uniform() < 0.5:  # scratch: distance to a segment of SCRATCH_LEN pixels
                    ang = rng.uniform(0, np.pi)
                    dy, dx = np.sin(ang), np.cos(ang)
                    half = (SCRATCH_LEN - 1) / 2
                    along = np.clip((yy - cy) * dy + (xx - cx) * dx, -half, half)
                    dist2 = (yy - cy - along * dy) ** 2 + (xx - cx - along * dx) ** 2
                    img += 1.3 * np.exp(-dist2 / (2 * 0.6**2))
                else:  # dent
                    img -= 1.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3**2))
            X[i, 0] = img + 0.25 * rng.normal(size=(SIZE, SIZE))
        return X, y

    X_train, y_train = build(n_train)
    X_test, y_test = build(n_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Helpers from earlier chapters (solved for you)
# ---------------------------------------------------------------------------
def conv_output_shape(size, kernel_size, stride=1, padding=0, dilation=1):
    """Output length along one axis."""
    reach = dilation * (kernel_size - 1) + 1
    return (size + 2 * padding - reach) // stride + 1


def col2im(dcol, x_shape, kernel_size, stride=1, padding=0):
    """Scatter-add columns back into an image: the transpose of im2col."""
    n, c, h, w_dim = x_shape
    k = kernel_size
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)
    dcol_r = dcol.reshape(n, c, k, k, h_out, w_out)
    canvas = np.zeros((n, c, h + 2 * padding, w_dim + 2 * padding), dtype=dcol.dtype)
    for dh in range(k):
        for dw in range(k):
            canvas[:, :, dh : dh + stride * h_out : stride, dw : dw + stride * w_out : stride] += dcol_r[:, :, dh, dw]
    if padding == 0:
        return canvas
    return canvas[:, :, padding:-padding, padding:-padding]


def relu(z):
    return np.maximum(z, 0.0)


def bce_with_logits(logits, y):
    """Mean binary cross-entropy and its gradient with respect to the logits."""
    loss = np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
    p = 0.5 * (1.0 + np.tanh(0.5 * logits))
    return float(loss), (p - y) / len(y)


# ---------------------------------------------------------------------------
# Stage 1: how far can a unit see?
# ---------------------------------------------------------------------------
def receptive_field(layers):
    """Receptive field of the last layer in a stack of (kernel_size, stride) layers."""
    r, jump = 1, 1
    for kernel_size, stride in layers:
        r += (kernel_size - 1) * jump
        jump *= stride
    return r


# ---------------------------------------------------------------------------
# Stage 2: convolution as one matrix multiply
# ---------------------------------------------------------------------------
def im2col(x, kernel_size, stride=1, padding=0):
    n, c, h, w_dim = x.shape
    k = kernel_size
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    col = np.zeros((n, c, k, k, h_out, w_out), dtype=x.dtype)
    for dh in range(k):
        for dw in range(k):
            col[:, :, dh, dw] = xp[:, :, dh : dh + stride * h_out : stride, dw : dw + stride * w_out : stride]
    return col.reshape(n, c * k * k, h_out * w_out)


def conv_forward(x, w, b, stride=1, padding=0):
    n, _, h, w_dim = x.shape
    c_out, _, k, _ = w.shape
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)
    col = im2col(x, k, stride, padding)
    out = np.einsum("oc,ncl->nol", w.reshape(c_out, -1), col) + b[None, :, None]
    return out.reshape(n, c_out, h_out, w_out)


def conv_backward(dY, x, w, stride=1, padding=0):
    n = x.shape[0]
    c_out, _, k, _ = w.shape
    col = im2col(x, k, stride, padding)
    dY_mat = dY.reshape(n, c_out, -1)
    db = dY_mat.sum(axis=(0, 2))
    dw = np.einsum("nol,ncl->oc", dY_mat, col).reshape(w.shape)
    dcol = np.einsum("oc,nol->ncl", w.reshape(c_out, -1), dY_mat)
    dx = col2im(dcol, x.shape, k, stride, padding)
    return dx, dw, db


# ---------------------------------------------------------------------------
# Stage 3: the residual block
# ---------------------------------------------------------------------------
def res_block_forward(x, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h1 = conv_forward(x, W1, b1, padding=1)
    a1 = relu(h1)
    h2 = conv_forward(a1, W2, b2, padding=1)
    y = x + h2
    return y, (x, W1, W2, h1, a1)


def res_block_backward(dY, cache):
    x, W1, W2, h1, a1 = cache
    da1, dW2, db2 = conv_backward(dY, a1, W2, padding=1)
    dh1 = da1 * (h1 > 0)
    dx_branch, dW1, db1 = conv_backward(dh1, x, W1, padding=1)
    dx = dY + dx_branch
    return dx, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------------------------------------------------------------------------
# Stage 4: pooling, head, and the whole network
# ---------------------------------------------------------------------------
def gap_forward(x):
    return x.mean(axis=(2, 3))


def gap_backward(dfeat, x_shape):
    n, c, h, w_dim = x_shape
    return np.broadcast_to(dfeat[:, :, None, None] / (h * w_dim), x_shape).copy()


def init_cnn(seed=0, channels=8):
    """Small random weights for the stem, the residual block, and the head (solved for you)."""
    rng = np.random.default_rng(seed)
    c = channels

    def he(shape):
        return rng.normal(size=shape) * np.sqrt(2.0 / np.prod(shape[1:]))

    return {
        "W0": he((c, 1, 3, 3)), "b0": np.zeros(c),
        "W1": he((c, c, 3, 3)) * 0.5, "b1": np.zeros(c),
        "W2": he((c, c, 3, 3)) * 0.5, "b2": np.zeros(c),
        "Wo": rng.normal(size=c) * 0.3, "bo": np.zeros(1),
    }


def cnn_forward(params, X):
    z0 = conv_forward(X, params["W0"], params["b0"], padding=1)
    a0 = relu(z0)
    y, rcache = res_block_forward(a0, params)
    a2 = relu(y)
    feat = gap_forward(a2)
    logits = feat @ params["Wo"] + params["bo"][0]
    return logits, (X, z0, a0, rcache, y, a2, feat)


def cnn_backward(params, cache, dlogits):
    X, z0, a0, rcache, y, a2, feat = cache
    grads = {"Wo": feat.T @ dlogits, "bo": np.array([dlogits.sum()])}
    dfeat = np.outer(dlogits, params["Wo"])
    da2 = gap_backward(dfeat, a2.shape)
    dy = da2 * (y > 0)
    da0, block_grads = res_block_backward(dy, rcache)
    grads.update(block_grads)
    dz0 = da0 * (z0 > 0)
    _, grads["W0"], grads["b0"] = conv_backward(dz0, X, params["W0"], padding=1)
    return grads


# ---------------------------------------------------------------------------
# Stage 5: optimiser, training loop, baseline
# ---------------------------------------------------------------------------
def adam_step(param, grad, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    state["m"] = beta1 * state["m"] + (1 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1 - beta2) * grad**2
    m_hat = state["m"] / (1 - beta1 ** state["t"])
    v_hat = state["v"] / (1 - beta2 ** state["t"])
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)


def fit(forward, backward, params, X, y, epochs=30, batch_size=50, lr=0.01, seed=0):
    """Train with mini-batch Adam and return (params, loss per epoch) (solved for you).

    `forward(params, X)` returns (logits, cache). `backward(params, cache,
    dlogits)` returns a dict of gradients with the keys of `params`. Your
    `adam_step` updates every array.
    """
    rng = np.random.default_rng(seed)
    params = {k: v.copy() for k, v in params.items()}
    states = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0} for k, v in params.items()}
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        total = 0.0
        for start in range(0, len(y), batch_size):
            idx = order[start : start + batch_size]
            logits, cache = forward(params, X[idx])
            loss, dlogits = bce_with_logits(logits, y[idx])
            grads = backward(params, cache, dlogits)
            for k in params:
                params[k] = adam_step(params[k], grads[k], states[k], lr=lr)
            total += loss * len(idx)
        history.append(total / len(y))
    return params, history


def init_mlp(seed=0, hidden=6):
    """A flat one-hidden-layer network on the raw pixels (solved for you)."""
    rng = np.random.default_rng(seed)
    d = SIZE * SIZE
    return {
        "W1": rng.normal(size=(d, hidden)) * np.sqrt(2.0 / d), "b1": np.zeros(hidden),
        "W2": rng.normal(size=hidden) * np.sqrt(1.0 / hidden), "b2": np.zeros(1),
    }


def mlp_forward(params, X):
    flat = X.reshape(len(X), -1)
    h = flat @ params["W1"] + params["b1"]
    a = relu(h)
    return a @ params["W2"] + params["b2"][0], (flat, h, a)


def mlp_backward(params, cache, dlogits):
    flat, h, a = cache
    dh = np.outer(dlogits, params["W2"]) * (h > 0)
    return {
        "W1": flat.T @ dh, "b1": dh.sum(axis=0),
        "W2": a.T @ dlogits, "b2": np.array([dlogits.sum()]),
    }


def count_params(params):
    return int(sum(v.size for v in params.values()))


def accuracy(forward, params, X, y):
    logits, _ = forward(params, X)
    return float(np.mean((logits > 0) == (y > 0.5)))


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = make_data(seed=0)
    print("train", X_tr.shape, "test", X_te.shape, "defect rate", y_tr.mean())

    stack = [(3, 1), (3, 1), (3, 1)]
    print("receptive field of the final layer:", receptive_field(stack), "pixels; a scratch spans", SCRATCH_LEN)

    cnn, cnn_loss = fit(cnn_forward, cnn_backward, init_cnn(0), X_tr, y_tr, epochs=20)
    mlp, mlp_loss = fit(mlp_forward, mlp_backward, init_mlp(0), X_tr, y_tr, epochs=20)
    print(f"CNN: {count_params(cnn)} parameters, test accuracy {accuracy(cnn_forward, cnn, X_te, y_te):.3f}")
    print(f"MLP: {count_params(mlp)} parameters, test accuracy {accuracy(mlp_forward, mlp, X_te, y_te):.3f}")

    shifted = np.roll(X_te, shift=(2, -3), axis=(2, 3))
    print(f"CNN accuracy after shifting every picture: {accuracy(cnn_forward, cnn, shifted, y_te):.3f}")
