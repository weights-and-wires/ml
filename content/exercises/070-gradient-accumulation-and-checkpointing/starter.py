"""Gradient accumulation and activation checkpointing.

Fill in the five functions marked `NotImplementedError`. The chain model and
`mean_loss_grads` are provided. The model is a stack of square layers
`h -> tanh(h @ W)`, and the loss is the mean over rows of `0.5 * sum((out - y) ** 2)`.
"""

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
    """Bytes to keep one layer-input array of shape `(batch, width)` for each layer.

    Returns an integer. Eight rows, width 4, three layers, fp32 gives 384.
    """
    raise NotImplementedError("write activation_bytes")


def micro_batch_weights(sizes):
    """Each micro-batch's share of all the examples.

    `sizes` is a list of micro-batch sizes. Returns a float array that sums to 1,
    so `[2, 3, 5]` gives `[0.2, 0.3, 0.5]`.
    """
    raise NotImplementedError("write micro_batch_weights")


def accumulate_grads(params, batches):
    """Gradient of the mean loss over every example in `batches`.

    `batches` is a list of `(x, y)` micro-batches whose sizes may differ.
    Call `mean_loss_grads` once per micro-batch, never on the joined data.
    Returns `(loss, grads)`: the float mean loss over all examples, and one
    gradient per weight matrix, equal to `mean_loss_grads` on the joined data.
    """
    raise NotImplementedError("write accumulate_grads")


def stored_peak(layers, segments):
    """Peak number of layer inputs held at once by checkpointing.

    Every segment boundary stays stored, and the backward pass for one segment
    also holds that segment's other inputs. Segments come from
    `np.array_split(np.arange(layers), segments)`, so the largest has
    `ceil(layers / segments)` layers and its first input is a boundary.
    Returns an integer. For 16 layers in 4 segments it is 7.
    """
    raise NotImplementedError("write stored_peak")


def best_segment_count(layers):
    """The segment count in `1..layers` with the smallest `stored_peak`.

    Ties go to the smaller count. Returns an integer. For 16 layers it is 4.
    """
    raise NotImplementedError("write best_segment_count")


def checkpointed_backward(params, x, y, segments):
    """Same loss and gradients as `mean_loss_grads`, storing only segment boundaries.

    Forward: split the layers with `np.array_split(np.arange(len(params)), segments)`,
    keep each segment's input array, and drop everything inside. Backward: for each
    segment from the last, rebuild its inputs with `layer_forward`, then call
    `layer_backward` from the last layer to the first. Call `layer_forward` by
    that name. A layer's last output is never rebuilt, because `layer_backward`
    only needs the layer's input.

    Returns `(loss, grads, stats)`. `stats["peak_stored"]` is the most layer
    inputs held at one time, and `stats["recomputed_layers"]` counts the
    `layer_forward` calls made in the backward pass.
    """
    raise NotImplementedError("write checkpointed_backward")


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
