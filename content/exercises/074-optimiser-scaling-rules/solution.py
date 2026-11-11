"""Optimiser scaling rules: muP learning rates and initialisation, a one-step
check on activations, a learning-rate sweep, and the Muon update.

Fill in the six functions below. The helpers
above them (`sp_init`, `loss_and_grads`, `adam_step`, and `train_loss`) are
written for you, because you built the same pieces in earlier chapters.
"""

import numpy as np

D_IN = 16
BASE_WIDTH = 32
LAYERS = ("input", "hidden", "output")
# Newton-Schulz coefficients tuned by Keller Jordan (2024) for the quintic step.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------- given
def sp_init(widths, rng) -> list:
    """Standard parameterisation: every weight matrix has std 1/sqrt(fan_in).

    Args:
        widths: Layer sizes [d_in, n, n, d_out].
        rng: A numpy Generator.

    Returns:
        A list of weight matrices, the i-th with shape (widths[i+1], widths[i]).
    """
    return [rng.normal(0.0, 1.0 / np.sqrt(widths[i]), (widths[i + 1], widths[i]))
            for i in range(len(widths) - 1)]


def loss_and_grads(Ws, x, y):
    """Forward and backward pass of a ReLU MLP with no biases and a linear output.

    Args:
        Ws: Weight matrices, the i-th with shape (fan_out, fan_in).
        x: Inputs, shape (batch, d_in).
        y: Targets, shape (batch, d_out).

    Returns:
        (loss, grads, pre): the mean of 0.5 * (out - y)**2, one gradient per
        weight matrix, and the pre-activation of every layer, each of shape
        (fan_out, batch).
    """
    acts, pre = [x.T], []
    a = x.T
    for i, W in enumerate(Ws):
        z = W @ a
        pre.append(z)
        a = z if i == len(Ws) - 1 else np.maximum(z, 0.0)
        acts.append(a)
    err = acts[-1] - y.T
    loss = 0.5 * np.mean(err**2)
    delta = err / x.shape[0]
    grads = [None] * len(Ws)
    for i in reversed(range(len(Ws))):
        grads[i] = delta @ acts[i].T
        if i > 0:
            delta = (Ws[i].T @ delta) * (pre[i - 1] > 0)
    return loss, grads, pre


def adam_step(Ws, grads, m, v, t, lrs, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam step with a separate learning rate for each weight matrix.

    Args:
        Ws, grads, m, v: Lists of arrays with matching shapes.
        t: The step number, starting at 1.
        lrs: One learning rate per matrix.

    Returns:
        New lists (Ws, m, v). The inputs are not changed.
    """
    new_W, new_m, new_v = [], [], []
    for W, g, mi, vi, lr in zip(Ws, grads, m, v, lrs):
        mi = beta1 * mi + (1 - beta1) * g
        vi = beta2 * vi + (1 - beta2) * g**2
        step = (mi / (1 - beta1**t)) / (np.sqrt(vi / (1 - beta2**t)) + eps)
        new_W.append(W - lr * step)
        new_m.append(mi)
        new_v.append(vi)
    return new_W, new_m, new_v


def train_loss(width, param, lr, seed, steps=200) -> float:
    """Train a [D_IN, width, width, 1] MLP on a fixed teacher and return the test loss.

    `param` is "sp" or "mup". For "mup" this calls your `mup_init` and your
    `adam_lr_for_width`. For "sp" every layer uses `lr`.
    """
    rng = np.random.default_rng(seed)
    widths = [D_IN, width, width, 1]
    if param == "mup":
        Ws = mup_init(widths, rng)
        lrs = [adam_lr_for_width(lr, BASE_WIDTH, width, layer) for layer in LAYERS]
    else:
        Ws = sp_init(widths, rng)
        lrs = [lr, lr, lr]
    t_rng = np.random.default_rng(123)
    T1 = t_rng.normal(size=(8, D_IN)) / np.sqrt(D_IN)
    T2 = t_rng.normal(size=(1, 8))
    m = [np.zeros_like(W) for W in Ws]
    v = [np.zeros_like(W) for W in Ws]
    d_rng = np.random.default_rng(7)
    for t in range(1, steps + 1):
        x = d_rng.normal(size=(64, D_IN))
        _, g, _ = loss_and_grads(Ws, x, np.tanh(x @ T1.T) @ T2.T)
        Ws, m, v = adam_step(Ws, g, m, v, t, lrs)
    xe = d_rng.normal(size=(512, D_IN))
    return float(loss_and_grads(Ws, xe, np.tanh(xe @ T1.T) @ T2.T)[0])


# ------------------------------------------------------------ to write
def adam_lr_for_width(base_lr, base_width, width, layer) -> float:
    """The muP Adam learning rate of one layer at a given width.

    The input layer keeps `base_lr`. A hidden layer and the output layer use
    `base_lr * base_width / width`.

    Args:
        base_lr: The rate tuned at `base_width`.
        base_width: The width of the small proxy model.
        width: The width of the model you are about to train.
        layer: "input", "hidden", or "output".

    Returns:
        The learning rate as a float. Raises ValueError for any other `layer`.
    """
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
    if layer == "input":
        return float(base_lr)
    return float(base_lr * base_width / width)


def mup_init(widths, rng) -> list:
    """Initialise a bias-free MLP under muP.

    The input and hidden matrices have std 1/sqrt(fan_in). The last matrix, the
    output layer, has std 1/fan_in.

    Args:
        widths: Layer sizes [d_in, n, n, d_out], at least three entries.
        rng: A numpy Generator. Draw the matrices in order, first layer first,
            with `rng.normal(0.0, std, (fan_out, fan_in))`.

    Returns:
        A list of len(widths) - 1 matrices, the i-th of shape (widths[i+1], widths[i]).
    """
    last = len(widths) - 2
    Ws = []
    for i in range(len(widths) - 1):
        fan_in, fan_out = widths[i], widths[i + 1]
        std = 1.0 / fan_in if i == last else 1.0 / np.sqrt(fan_in)
        Ws.append(rng.normal(0.0, std, (fan_out, fan_in)))
    return Ws


def activation_update_size(width, param, seed=0, lr=1e-3) -> float:
    """RMS change in the second hidden layer's pre-activation after one Adam step.

    Build a [D_IN, width, width, 1] network with `sp_init` (param "sp") or
    `mup_init` (param "mup"), draw x of shape (8, D_IN) and y of shape (8, 1)
    from the same generator after the weights, take one Adam step at t = 1 with
    zero moments, and return the root mean square of the change in pre[1].
    For "sp" every layer uses `lr`. For "mup" use `adam_lr_for_width` with
    BASE_WIDTH.

    Returns:
        A float.
    """
    rng = np.random.default_rng(seed)
    widths = [D_IN, width, width, 1]
    if param == "mup":
        Ws = mup_init(widths, rng)
        lrs = [adam_lr_for_width(lr, BASE_WIDTH, width, layer) for layer in LAYERS]
    else:
        Ws = sp_init(widths, rng)
        lrs = [lr, lr, lr]
    x = rng.normal(size=(8, D_IN))
    y = rng.normal(size=(8, 1))
    _, grads, pre = loss_and_grads(Ws, x, y)
    zeros = [np.zeros_like(W) for W in Ws]
    new_Ws, _, _ = adam_step(Ws, grads, zeros, zeros, 1, lrs)
    _, _, pre_new = loss_and_grads(new_Ws, x, y)
    return float(np.sqrt(np.mean((pre_new[1] - pre[1]) ** 2)))


def best_lr(width, param, lr_grid, seeds=(0, 1, 2, 3)) -> float:
    """The learning rate with the lowest mean final loss.

    For every rate in `lr_grid`, average `train_loss(width, param, lr, seed)`
    over `seeds`. Treat a non-finite loss as 1e9. Return the rate with the
    smallest average.
    """
    means = []
    for lr in lr_grid:
        losses = [train_loss(width, param, lr, s) for s in seeds]
        losses = [x if np.isfinite(x) else 1e9 for x in losses]
        means.append(np.mean(losses))
    return float(lr_grid[int(np.argmin(means))])


def newton_schulz(G, steps=5) -> np.ndarray:
    """Push every singular value of G toward 1 and keep its singular vectors.

    Divide G by its Frobenius norm plus 1e-7. If the matrix has more rows than
    columns, work on its transpose. Then repeat `steps` times, with
    S = X @ X.T:

        X = a * X + b * (S @ X) + c * (S @ S @ X)

    where (a, b, c) = NS_COEFFS. Undo the transpose at the end.

    Args:
        G: A matrix of shape (m, n).
        steps: The number of iterations.

    Returns:
        A matrix of shape (m, n) whose singular values lie near 1.
    """
    a, b, c = NS_COEFFS
    X = np.asarray(G, dtype=float)
    X = X / (np.linalg.norm(X) + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        S = X @ X.T
        X = a * X + b * (S @ X) + c * (S @ S @ X)
    return X.T if tall else X


def muon_step(W, G, M, lr, beta=0.95) -> tuple:
    """One Muon step: momentum, then an orthogonalised update.

    Args:
        W: Weights, shape (m, n).
        G: Gradient of the loss with respect to W, shape (m, n).
        M: Momentum buffer, shape (m, n).
        lr: Learning rate.
        beta: Momentum coefficient.

    Returns:
        (W_new, M_new), where M_new = beta * M + G and
        W_new = W - lr * newton_schulz(M_new).
    """
    M_new = beta * M + G
    return W - lr * newton_schulz(M_new), M_new
