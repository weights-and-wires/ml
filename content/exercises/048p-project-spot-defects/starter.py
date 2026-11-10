"""Project: spot defects on a production line with a small residual CNN.

Fill in every function that raises NotImplementedError. Run this file to train
the network once your functions work.
"""

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
    """Receptive field of the last layer in a stack of convolution layers.

    Args:
        layers: List of (kernel_size, stride) pairs, first layer first.

    Returns:
        Integer: how many input pixels, along one axis, the last layer's
        unit depends on. `[(3, 1)]` gives 3 and `[(3, 2), (3, 1)]` gives 7.

    Steps:
    1. Start with a field `r = 1` and a jump `j = 1`, the distance in input pixels between neighbouring units.
    2. For each layer, add `(kernel_size - 1) * j` to `r`, then multiply `j` by the layer's stride. Update `r` before `j`.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: convolution as one matrix multiply
# ---------------------------------------------------------------------------
def im2col(x, kernel_size, stride=1, padding=0):
    """Lay every window of x out as a column.

    Args:
        x: Input, shape (N, C, H, W).
        kernel_size: Side length k of the square window.
        stride: Step between windows.
        padding: Zeros added to each spatial side first.

    Returns:
        Array of shape (N, C * k * k, H_out * W_out). Column l holds the window
        for output position (l // W_out, l % W_out), flattened in (channel,
        kernel row, kernel column) order so it lines up with
        `w.reshape(C_out, -1)`.

    Steps:
    1. Pad `x` with `np.pad`, then allocate `col` with shape `(N, C, k, k, H_out, W_out)`.
    2. Loop over the k * k kernel offsets only. For offset (dh, dw), a strided slice of the padded image gives that offset's value at every output position at once.
    3. Reshape `col` to `(N, C * k * k, H_out * W_out)`.
    """
    raise NotImplementedError


def conv_forward(x, w, b, stride=1, padding=0):
    """Convolution as one matrix multiply on the output of `im2col`.

    Args:
        x: Input, shape (N, C_in, H, W).
        w: Kernels, shape (C_out, C_in, k, k).
        b: Biases, shape (C_out,).

    Returns:
        Output, shape (N, C_out, H_out, W_out). No Python loop over positions,
        channels, or images.

    Steps:
    1. Build `col = im2col(x, k, stride, padding)`.
    2. Multiply `w.reshape(C_out, -1)` into every image's columns, for example with `np.einsum('oc,ncl->nol', ...)`, and add `b[None, :, None]`.
    3. Reshape to `(N, C_out, H_out, W_out)` using `conv_output_shape`.
    """
    raise NotImplementedError


def conv_backward(dY, x, w, stride=1, padding=0):
    """Backward pass of `conv_forward`.

    Args:
        dY: Gradient with respect to the output, shape (N, C_out, H_out, W_out).
        x: The forward input, shape (N, C_in, H, W).
        w: The forward kernels, shape (C_out, C_in, k, k).

    Returns:
        (dx, dw, db) shaped like (x, w, b). Gradients for w and b add up over
        every image and every position.

    Steps:
    1. Rebuild `col = im2col(x, ...)` and flatten `dY` to `(N, C_out, L)`.
    2. `db` sums `dY` over images and positions. `dw` is the product of `dY` with `col`, summed over images, reshaped to `w.shape`.
    3. `dcol` is `w.reshape(C_out, -1).T` times `dY`; hand it to `col2im` to get `dx`.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: the residual block
# ---------------------------------------------------------------------------
def res_block_forward(x, params):
    """Forward pass of y = x + conv(relu(conv(x))), both convolutions 3 by 3 with padding 1.

    Args:
        x: Input, shape (N, C, H, W).
        params: Dict with W1, W2 of shape (C, C, 3, 3) and b1, b2 of shape (C,).

    Returns:
        (y, cache). `y` has the shape of `x`. `cache` is whatever
        `res_block_backward` needs.

    Steps:
    1. First convolution, ReLU, second convolution, all with `padding=1` so the size stays the same.
    2. Add the input `x` to the second convolution's output. That addition is the shortcut.
    """
    raise NotImplementedError


def res_block_backward(dY, cache):
    """Backward pass of `res_block_forward`.

    Args:
        dY: Gradient with respect to y, shape (N, C, H, W).
        cache: What `res_block_forward` returned.

    Returns:
        (dx, grads). `grads` has keys W1, b1, W2, b2 shaped like the parameters.

    Steps:
    1. Send `dY` back through the second convolution, then through the ReLU mask `h1 > 0`, then through the first convolution.
    2. The shortcut hands `dY` straight to the input, so `dx` is `dY` plus the branch's contribution.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: pooling, head, and the whole network
# ---------------------------------------------------------------------------
def gap_forward(x):
    """Global average pooling.

    Args:
        x: Feature maps, shape (N, C, H, W).

    Returns:
        Array of shape (N, C): the mean of each channel over all positions.

    Steps:
    1. Average over the two spatial axes.
    """
    raise NotImplementedError


def gap_backward(dfeat, x_shape):
    """Backward pass of `gap_forward`.

    Args:
        dfeat: Gradient with respect to the pooled features, shape (N, C).
        x_shape: The (N, C, H, W) shape of the input to `gap_forward`.

    Returns:
        Array of shape x_shape.

    Steps:
    1. Every position contributed `1 / (H * W)` of the mean, so each one receives `dfeat / (H * W)`.
    2. Broadcast the result over the two spatial axes.
    """
    raise NotImplementedError


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
    """Forward pass of the whole defect detector.

    The layers are: 3 by 3 stem convolution (padding 1) and ReLU, the residual
    block, ReLU, global average pooling, then a linear head.

    Args:
        params: Dict from `init_cnn`: stem W0, b0; block W1, b1, W2, b2; head
            Wo of shape (C,) and bo of shape (1,).
        X: Pictures, shape (N, 1, H, W).

    Returns:
        (logits, cache). `logits` has shape (N,): positive means defective.

    Steps:
    1. Stem convolution and ReLU, then `res_block_forward` (it reads W1, b1, W2, b2 from the same dict).
    2. ReLU, `gap_forward`, then `feat @ Wo + bo[0]`.
    3. Keep every intermediate the backward pass needs in `cache`.
    """
    raise NotImplementedError


def cnn_backward(params, cache, dlogits):
    """Backward pass of `cnn_forward`.

    Args:
        params: The parameter dict.
        cache: What `cnn_forward` returned.
        dlogits: Gradient with respect to the logits, shape (N,).

    Returns:
        Dict of gradients with exactly the keys of `params`, each shaped like
        its parameter.

    Steps:
    1. Head: `Wo` gets `feat.T @ dlogits`, `bo` gets the sum of `dlogits` (as a shape (1,) array), and `dfeat` is the outer product of `dlogits` and `Wo`.
    2. `gap_backward`, then the mask of the ReLU after the block, then `res_block_backward`.
    3. Mask the ReLU after the stem and call `conv_backward` for `W0` and `b0`.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: optimiser, training loop, baseline
# ---------------------------------------------------------------------------
def adam_step(param, grad, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam update for one array.

    Args:
        param: Current values.
        grad: Gradient, same shape.
        state: Dict with arrays `m` and `v` (same shape as `param`) and an
            integer `t`. Update it in place.
        lr, beta1, beta2, eps: Adam settings.

    Returns:
        The new values of `param`.

    Steps:
    1. Add 1 to `state['t']`, then update `m` and `v` as exponential averages of `grad` and `grad**2`.
    2. Correct the bias: divide `m` by `1 - beta1**t` and `v` by `1 - beta2**t`.
    3. Return `param - lr * m_hat / (sqrt(v_hat) + eps)`.
    """
    raise NotImplementedError


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
