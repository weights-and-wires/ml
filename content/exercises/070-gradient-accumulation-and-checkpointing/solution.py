"""Reference solution for the gradient accumulation and checkpointing exercise."""

from __future__ import annotations

import math

import numpy as np


def layer_forward(W, h):
    """Provided. One layer of the chain: `tanh(h @ W)`."""
    return np.tanh(h @ W)


def layer_backward(W, h, dout):
    """Provided. Backward through one layer, given the layer's input `h`.

    The layer's own output is rebuilt from `h` inside this function, so you
    only have to keep the input. Returns `(dh, dW)`.
    """
    a = np.tanh(h @ W)
    dz = dout * (1.0 - a * a)
    return dz @ W.T, h.T @ dz


def init_chain(layers, width, seed=0):
    """Provided. A list of `layers` square weight matrices of size `width`."""
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(width, width)) / math.sqrt(width) for _ in range(layers)]


def mean_loss_grads(params, x, y):
    """Provided. Plain backprop that stores every layer input.

    The loss is the mean over the rows of `0.5 * sum((out - y) ** 2)`.
    Returns `(loss, grads)` with one gradient per weight matrix.
    """
    hs = [x]
    for W in params:
        hs.append(layer_forward(W, hs[-1]))
    diff = hs[-1] - y
    loss = 0.5 * float(np.sum(diff * diff)) / len(x)
    dout = diff / len(x)
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        dout, grads[i] = layer_backward(params[i], hs[i], dout)
    return loss, grads


def activation_bytes(batch, width, layers, bytes_per_number=4):
    """Bytes to keep one layer-input array of `(batch, width)` for each layer."""
    return batch * width * layers * bytes_per_number


def micro_batch_weights(sizes):
    """Each micro-batch's share of all the examples, as a float array."""
    sizes = np.asarray(sizes, dtype=float)
    return sizes / sizes.sum()


def accumulate_grads(params, batches):
    """Gradient of the mean loss over every example in `batches`.

    `batches` is a list of `(x, y)` micro-batches whose sizes may differ.
    Returns `(loss, grads)` equal to `mean_loss_grads` on the joined data.
    """
    weights = micro_batch_weights([len(x) for x, _ in batches])
    total = 0.0
    acc = [np.zeros_like(W) for W in params]
    for w, (x, y) in zip(weights, batches):
        loss, grads = mean_loss_grads(params, x, y)
        total += w * loss
        for a, g in zip(acc, grads):
            a += w * g
    return total, acc


def stored_peak(layers, segments):
    """Peak layer inputs held: every boundary plus the largest segment's interior."""
    return segments + math.ceil(layers / segments) - 1


def best_segment_count(layers):
    """The segment count in 1..layers with the smallest `stored_peak`; ties go to fewer."""
    return min(range(1, layers + 1), key=lambda k: (stored_peak(layers, k), k))


def checkpointed_backward(params, x, y, segments):
    """Same loss and gradients as `mean_loss_grads`, storing only segment boundaries.

    Returns `(loss, grads, stats)` where `stats` holds `peak_stored` and
    `recomputed_layers`.
    """
    L = len(params)
    cuts = np.array_split(np.arange(L), segments)
    held = []
    peak = 0
    h = x
    boundaries = []
    for idx in cuts:
        boundaries.append(h)
        held.append(h)
        peak = max(peak, len(held))
        for i in idx:
            h = layer_forward(params[i], h)
    diff = h - y
    loss = 0.5 * float(np.sum(diff * diff)) / len(x)
    dout = diff / len(x)
    grads = [None] * L
    recomputed = 0
    for s in reversed(range(len(cuts))):
        idx = cuts[s]
        inputs = [boundaries[s]]
        for i in idx[:-1]:
            inputs.append(layer_forward(params[i], inputs[-1]))
            held.append(inputs[-1])
            recomputed += 1
            peak = max(peak, len(held))
        for pos in reversed(range(len(idx))):
            dout, grads[idx[pos]] = layer_backward(params[idx[pos]], inputs[pos], dout)
        del held[len(held) - (len(idx) - 1):]
    return loss, grads, {"peak_stored": peak, "recomputed_layers": recomputed}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    params = init_chain(16, 8)
    x, y = rng.normal(size=(12, 8)), rng.normal(size=(12, 8))
    loss, g = mean_loss_grads(params, x, y)
    batches = [(x[:3], y[:3]), (x[3:7], y[3:7]), (x[7:], y[7:])]
    loss_a, g_a = accumulate_grads(params, batches)
    print("full-batch loss", round(loss, 6), "accumulated loss", round(loss_a, 6))
    print("largest gradient gap", max(float(np.abs(a - b).max()) for a, b in zip(g, g_a)))
    k = best_segment_count(16)
    loss_c, g_c, stats = checkpointed_backward(params, x, y, k)
    print("best segments for 16 layers:", k)
    print("checkpoint gap", max(float(np.abs(a - b).max()) for a, b in zip(g, g_c)), stats)
    print("activation bytes, plain:", activation_bytes(12, 8, 16))
