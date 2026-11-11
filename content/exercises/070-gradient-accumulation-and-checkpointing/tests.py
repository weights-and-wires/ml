"""Checks for the gradient accumulation and checkpointing exercise.

Plausible wrong implementations these checks catch:

* accumulate_grads that adds the micro-batch mean gradients without dividing by
  the number of micro-batches. The gradient is k times too large. The equal-size
  check catches it.
* accumulate_grads that averages the micro-batch means with weight 1/k. It is
  right for equal sizes and wrong for unequal ones. The unequal-size check
  catches it.
* checkpointed_backward that stores nothing and rebuilds from the input each
  time, or that keeps every activation. The layer_forward call count and the
  peak check catch them.
"""

import math

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has


def _data(n, width, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, width)), rng.normal(size=(n, width))


def _split(x, y, sizes):
    out, start = [], 0
    for s in sizes:
        out.append((x[start:start + s], y[start:start + s]))
        start += s
    return out


def _loss(params, x, y):
    h = x
    for W in params:
        h = np.tanh(h @ W)
    return 0.5 * float(np.sum((h - y) ** 2)) / len(x)


@check("activation_bytes counts one (batch, width) array per layer (worked: 384 bytes)")
def test_activation_bytes(sub):
    fn = has(sub, "activation_bytes")
    equal(int(fn(8, 4, 3)), 384, label="8 rows, width 4, 3 layers, fp32")
    equal(int(fn(16, 4, 3)), 768, label="doubling the batch")
    equal(int(fn(8, 4, 6, bytes_per_number=2)), 384, label="6 layers in fp16")


@check("micro_batch_weights gives each micro-batch its share of the examples")
def test_weights(sub):
    fn = has(sub, "micro_batch_weights")
    close(fn([2, 3, 5]), np.array([0.2, 0.3, 0.5]), atol=1e-12, label="shares of [2, 3, 5]")
    close(float(np.sum(fn([1, 2, 5, 8]))), 1.0, atol=1e-12, label="the shares sum to 1")


@check("stored_peak counts boundaries plus the largest segment (16 layers: 7, 16, 16)")
def test_stored_peak(sub):
    fn = has(sub, "stored_peak")
    equal(int(fn(16, 4)), 7, label="16 layers in 4 segments")
    equal(int(fn(16, 1)), 16, label="one segment keeps everything")
    equal(int(fn(16, 16)), 16, label="one layer per segment keeps everything")
    equal(int(fn(10, 4)), 6, label="10 layers in 4 segments, largest has 3")


@check("best_segment_count minimises the stored peak and keeps the peak under 2 sqrt(L) + 1")
def test_best_segments(sub):
    fn = has(sub, "best_segment_count")
    peak = has(sub, "stored_peak")
    equal(int(fn(16)), 4, label="best count for 16 layers")
    equal(int(fn(100)), 10, label="best count for 100 layers")
    equal(int(fn(1)), 1, label="best count for 1 layer")
    for L in range(1, 61):
        k = int(fn(L))
        best = min(peak(L, j) for j in range(1, L + 1))
        if peak(L, k) != best:
            fail(f"L={L}: best_segment_count gave {k} with peak {peak(L, k)}, the smallest peak is {best}")
        if peak(L, k) > 2 * math.sqrt(L) + 1:
            fail(f"L={L}: peak {peak(L, k)} is above 2 sqrt(L) + 1 = {2 * math.sqrt(L) + 1:.2f}")


@check("accumulate_grads over equal micro-batches equals the full-batch gradient to 1e-10")
def test_accumulate_equal(sub):
    fn = has(sub, "accumulate_grads")
    params = sub.init_chain(4, 5, seed=1)
    x, y = _data(12, 5, seed=2)
    loss, grads = fn(params, _split(x, y, [3, 3, 3, 3]))
    want_loss, want = sub.mean_loss_grads(params, x, y)
    almost(loss, want_loss, 1e-10, label="accumulated loss")
    for i, (g, w) in enumerate(zip(grads, want)):
        close(g, w, atol=1e-10, rtol=0, label=f"gradient of layer {i} (4 micro-batches of 3)")
    f = lambda v: _loss([v.reshape(5, 5)] + params[1:], x, y)
    close(grads[0], finite_difference(f, params[0].ravel().copy()).reshape(5, 5), atol=1e-6, label="layer 0 against finite differences")


@check("accumulate_grads with unequal micro-batches still equals the full batch (sizes 1, 2, 5, 8)")
def test_accumulate_unequal(sub):
    fn = has(sub, "accumulate_grads")
    params = sub.init_chain(3, 4, seed=3)
    x, y = _data(16, 4, seed=4)
    loss, grads = fn(params, _split(x, y, [1, 2, 5, 8]))
    want_loss, want = sub.mean_loss_grads(params, x, y)
    almost(loss, want_loss, 1e-10, label="accumulated loss")
    for i, (g, w) in enumerate(zip(grads, want)):
        close(g, w, atol=1e-10, rtol=0, label=f"gradient of layer {i}; weights must be n_i / N, not 1/k")


@check("checkpointed_backward returns the plain gradients for 1 segment, a split that does not divide, and L segments")
def test_checkpoint_grads(sub):
    fn = has(sub, "checkpointed_backward")
    params = sub.init_chain(10, 4, seed=5)
    x, y = _data(7, 4, seed=6)
    want_loss, want = sub.mean_loss_grads(params, x, y)
    for k in (1, 3, 4, 10):
        loss, grads, _ = fn(params, x, y, k)
        almost(loss, want_loss, 1e-12, label=f"loss with {k} segments")
        for i, (g, w) in enumerate(zip(grads, want)):
            close(g, w, atol=1e-10, rtol=0, label=f"gradient of layer {i} with {k} segments")


@check("checkpointing cuts the peak from 16 to 7 and rebuilds L - k layers (counted two ways)")
def test_checkpoint_counts(sub):
    fn = has(sub, "checkpointed_backward")
    params = sub.init_chain(16, 4, seed=7)
    x, y = _data(5, 4, seed=8)
    calls = [0]
    real = sub.layer_forward

    def counting(W, h):
        calls[0] += 1
        return real(W, h)

    sub.layer_forward = counting
    try:
        _, _, stats = fn(params, x, y, 4)
    finally:
        sub.layer_forward = real
    equal(int(stats["peak_stored"]), 7, label="peak layer inputs held, 16 layers in 4 segments")
    equal(int(stats["recomputed_layers"]), 12, label="reported recomputed layers (L - k)")
    equal(calls[0], 16 + 12, label="layer_forward calls: 16 forward plus 12 rebuilds (call layer_forward by name)")
    k = sub.best_segment_count(16)
    _, _, best = fn(params, x, y, k)
    if best["peak_stored"] > 2 * math.sqrt(16):
        fail(f"peak {best['peak_stored']} exceeds 2 sqrt(L) = {2 * math.sqrt(16)}")
