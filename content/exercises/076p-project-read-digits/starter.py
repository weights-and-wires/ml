"""Project: train a digit reader from scratch, in numpy only.

A network is a list of layers. Each layer is a dict {"W": array (n_in, n_out), "b": array (n_out,)},
so a layer computes `h @ W + b`. Every layer but the last is followed by a ReLU, and dropout
acts on the output of each ReLU. `make_data`, the glyph renderer, and `adamw_step` are written
for you. Fill in the other functions below, in order.
"""

import numpy as np

SIZE = 10
NUM_CLASSES = 10

# Ten stroke templates. Each digit is a list of (x, y) corners in a unit square,
# x to the right and y downward; the renderer joins the corners with line segments.
STROKES = {
    0: [(.25, .1), (.75, .1), (.9, .3), (.9, .7), (.75, .9), (.25, .9), (.1, .7), (.1, .3), (.25, .1)],
    1: [(.3, .28), (.55, .1), (.55, .9)],
    2: [(.15, .25), (.3, .1), (.7, .1), (.85, .25), (.8, .45), (.15, .9), (.9, .9)],
    3: [(.15, .15), (.8, .15), (.5, .45), (.85, .65), (.75, .9), (.15, .85)],
    4: [(.7, .9), (.7, .1), (.1, .65), (.9, .65)],
    5: [(.85, .1), (.2, .1), (.15, .45), (.7, .45), (.85, .65), (.7, .9), (.15, .85)],
    6: [(.8, .1), (.3, .3), (.15, .6), (.3, .9), (.7, .9), (.85, .7), (.7, .5), (.2, .55)],
    7: [(.1, .1), (.9, .1), (.4, .9)],
    8: [(.5, .1), (.2, .22), (.5, .48), (.8, .75), (.5, .9), (.2, .75), (.5, .48), (.8, .22), (.5, .1)],
    9: [(.85, .45), (.5, .55), (.2, .4), (.35, .12), (.7, .12), (.85, .3), (.8, .9)],
}


def _distance_to_polyline(px, py, corners):
    """Distance from every pixel centre (px, py) to the nearest segment of a polyline."""
    best = np.full(px.shape, np.inf)
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        dx, dy = x1 - x0, y1 - y0
        t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        best = np.minimum(best, np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))
    return best


def render_digit(digit, rng, noise=0.25):
    """Draw one noisy 10 x 10 glyph of `digit` and return it as a flat array of 100 pixels."""
    centres = (np.arange(SIZE) + 0.5) / SIZE
    px, py = np.meshgrid(centres, centres)
    corners = [(x + rng.normal(scale=0.03), y + rng.normal(scale=0.03)) for x, y in STROKES[digit]]
    shift = rng.integers(-1, 2, size=2) / SIZE
    corners = [(x * 0.9 + 0.05 + shift[0], y * 0.9 + 0.05 + shift[1]) for x, y in corners]
    thickness = rng.uniform(0.07, 0.15)
    ink = np.clip(1.5 - _distance_to_polyline(px, py, corners) / thickness, 0.0, 1.0)
    return (ink + noise * rng.normal(size=ink.shape)).ravel()


def make_data(seed=0, n_train=1500, n_val=500, n_test=1000):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) of noisy digit glyphs."""
    rng = np.random.default_rng(seed)

    def split(n):
        y = np.arange(n) % NUM_CLASSES
        rng.shuffle(y)
        return np.stack([render_digit(d, rng) for d in y]), y

    X_train, y_train = split(n_train)
    X_val, y_val = split(n_val)
    X_test, y_test = split(n_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def he_init(layer_sizes, rng):
    """Return He-initialised parameters for a network with the given layer sizes.

    Args:
        layer_sizes: A list such as [100, 64, 32, 10]: the input size, each hidden size, then the
            number of classes.
        rng: A `np.random.Generator`.

    Returns:
        A list with one dict per layer. "W" has shape (n_in, n_out) and holds normal draws with
        standard deviation sqrt(2 / n_in). "b" has shape (n_out,) and is all zeros.

    Steps:
        1. Pair each size with the next one.
        2. Draw W with `rng.normal(scale=..., size=(n_in, n_out))`.
    """
    raise NotImplementedError("he_init")


def forward(params, X, p_drop=0.0, rng=None, training=False):
    """Run the network forward and keep what the backward pass needs.

    Args:
        params: The list of layers.
        X: Inputs, shape (n, n_in).
        p_drop: Probability of dropping each hidden unit. Used only when `training` is True.
        rng: A `np.random.Generator` that draws the dropout masks. May be None when `training`
            is False.
        training: When False, no dropout happens and `rng` is never touched.

    Returns:
        `(logits, cache)`. `logits` has shape (n, n_classes) and holds the last layer's output
        with no activation. `cache` is a list with one dict per layer holding "h_in" (the layer's
        input), "z" (its pre-activation), and "mask" (the dropout multiplier applied to the
        ReLU output, already divided by 1 - p_drop, or None when there was no dropout).

    Steps:
        1. For each layer compute z = h @ W + b.
        2. For the last layer, return z as the logits. It gets no ReLU and no dropout.
        3. Otherwise apply ReLU, then in training mode multiply by a mask of
           `(rng.random(shape) >= p_drop) / (1 - p_drop)`.
        4. Draw masks one layer at a time, from the first hidden layer to the last, so that a
           generator with the same seed produces the same masks.
    """
    raise NotImplementedError("forward")


def smoothed_cross_entropy(logits, y, eps):
    """Return the mean label-smoothed cross-entropy and its gradient on the logits.

    Args:
        logits: Shape (n, k).
        y: Integer labels, shape (n,).
        eps: Smoothing amount. The target is (1 - eps) * onehot(y) + eps / k.

    Returns:
        `(loss, dlogits)`: a float, and an array of shape (n, k) equal to (softmax - target) / n.

    Steps:
        1. Compute a log-softmax with the row maximum subtracted first.
        2. Build the smoothed target matrix.
        3. The loss is the mean over rows of -(target * log_softmax).sum.
    """
    raise NotImplementedError("smoothed_cross_entropy")


def backward(params, cache, dlogits):
    """Backpropagate `dlogits` through the whole network.

    Args:
        params: The list of layers used in the forward pass.
        cache: The cache that `forward` returned.
        dlogits: Gradient of the loss on the logits, shape (n, k).

    Returns:
        A list with one dict per layer, {"W": ..., "b": ...}, with the same shapes as `params`.

    Steps:
        1. Start at the last layer. Its W gradient is h_in.T @ d and its b gradient is d.sum(axis=0).
        2. Move to the layer before it: d = d @ W.T, multiply by that layer's dropout mask if it
           has one, then by (z > 0) for the ReLU.
        3. Repeat until the first layer.
    """
    raise NotImplementedError("backward")


def loss_and_grads(params, X, y, eps=0.0, p_drop=0.0, rng=None, training=True):
    """Return the smoothed loss and the gradient of every parameter.

    Args:
        params: The list of layers.
        X: Inputs, shape (n, n_in).
        y: Integer labels, shape (n,).
        eps: Label-smoothing amount.
        p_drop: Dropout probability. Applied only when `training` is True.
        rng: Generator for the dropout masks.
        training: Whether dropout is on.

    Returns:
        `(loss, grads)` where `grads` has the same structure as `params`.
    """
    raise NotImplementedError("loss_and_grads")


def gradient_check(params, X, y, eps=0.0, grads=None, h=1e-5):
    """Return the worst relative error between backprop and finite differences.

    Args:
        params: The list of layers.
        X: Inputs, shape (n, n_in).
        y: Integer labels, shape (n,).
        eps: Label-smoothing amount used in the loss.
        grads: Gradients to test. When None, compute them with `loss_and_grads` and no dropout.
        h: Nudge size for the central difference.

    Returns:
        The largest value, over every "W" and "b" array, of
        ||g_backprop - g_numeric|| / (||g_backprop|| + ||g_numeric||). Treat 0 / 0 as 0.

    Steps:
        1. Nudge each entry of each array up and down by h, recompute the loss with no dropout,
           and divide the difference by 2h. Put the entry back afterwards.
        2. Compare the whole array at once with the relative error above.
    """
    raise NotImplementedError("gradient_check")


# Written in the chapter "AdamW: decaying weights the right way".
def adamw_step(w, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    t = state.get("t", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(w)) + (1.0 - beta1) * grad
    v = beta2 * state.get("v", np.zeros_like(w)) + (1.0 - beta2) * grad**2
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * w
    return new_w, {"m": m, "v": v, "t": t}


def adamw_update(params, grads, states, lr, weight_decay=0.01):
    """Apply one AdamW step to every array in the network.

    Args:
        params: The list of layers.
        grads: Gradients with the same structure.
        states: A list of {"W": state, "b": state} dicts, where each state is the dict that
            `adamw_step` takes and returns. Start from empty dicts.
        lr: The learning rate for this step.
        weight_decay: Decay for the weight matrices. Biases get no decay.

    Returns:
        `(new_params, new_states)`, built fresh. The inputs are not changed.

    Steps:
        1. Call `adamw_step` for each "W" with `weight_decay`.
        2. Call it for each "b" with `weight_decay=0.0`.
    """
    raise NotImplementedError("adamw_update")


def lr_schedule(step, warmup, total, peak):
    """Return the learning rate at `step`: linear warmup, then cosine decay to zero.

    Args:
        step: The step number, starting at 0.
        warmup: The number of warmup steps.
        total: The total number of steps.
        peak: The learning rate at the end of warmup.

    Returns:
        `peak * step / warmup` while `step < warmup`. After that,
        `peak * 0.5 * (1 + cos(pi * progress))`, where progress runs from 0 at the end of warmup
        to 1 at `total`.
    """
    raise NotImplementedError("lr_schedule")


def predict(params, X):
    """Return the predicted digit for each row of X, shape (n,), with dropout off."""
    raise NotImplementedError("predict")


def train(data, hidden=(64, 32), epochs=60, batch_size=64, peak_lr=3e-3, weight_decay=0.01,
          p_drop=0.2, smoothing=0.1, patience=10, seed=0):
    """Train a network with mini-batch AdamW and return the best parameters.

    Args:
        data: The tuple from `make_data`. Only the first four entries are used.
        hidden: Hidden layer sizes. An empty tuple gives a linear softmax model.
        epochs: The most epochs to run.
        batch_size: Examples per mini-batch.
        peak_lr: The peak of the warmup-plus-cosine schedule.
        weight_decay: AdamW decay on the weight matrices.
        p_drop: Dropout probability during training.
        smoothing: Label-smoothing amount for the training loss.
        patience: Stop when this many epochs in a row fail to beat the best validation loss.
        seed: Seeds one generator that draws the initial weights, the shuffles, and the masks.

    Returns:
        `(best_params, history)`. `best_params` are the parameters from the epoch with the lowest
        validation loss. `history` is a dict with one entry per epoch run: "train_loss" (mean of
        the training batches' loss), "val_loss" (plain cross-entropy, `eps=0`, with dropout off),
        "val_acc", and "lr" (the rate used on the epoch's last step). It also holds "best_epoch",
        the index of the best epoch.

    Steps:
        1. Build the parameters with `he_init` and an empty AdamW state for every array.
        2. Set total steps to epochs * ceil(n / batch_size) and warmup to 5 percent of that.
        3. Each epoch, shuffle with `rng.permutation`, then loop over mini-batches: get the rate
           from `lr_schedule`, `loss_and_grads` with dropout on, then `adamw_update`.
        4. After each epoch, measure validation loss and accuracy with dropout off. Copy the
           parameters when the loss is a new best. Stop once `patience` epochs have passed
           since the best.
    """
    raise NotImplementedError("train")


if __name__ == "__main__":
    import time

    data = make_data(0)
    X_train, y_train, X_val, y_val, X_test, y_test = data
    print("data:", X_train.shape, X_val.shape, X_test.shape)

    tiny = he_init([100, 8, 10], np.random.default_rng(0))
    print("gradient check relative error:", gradient_check(tiny, X_train[:20], y_train[:20], eps=0.1))

    start = time.time()
    params, history = train(data)
    print(f"epochs run {len(history['val_loss'])}, best epoch {history['best_epoch']}")
    print(f"MLP test accuracy {(predict(params, X_test) == y_test).mean():.3f} in {time.time() - start:.1f}s")
    linear, _ = train(data, hidden=(), p_drop=0.0)
    print(f"linear baseline {(predict(linear, X_test) == y_test).mean():.3f}")
