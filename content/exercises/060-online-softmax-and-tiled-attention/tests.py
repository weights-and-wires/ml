"""Checks for the online softmax and tiled attention exercise."""

import math
import tracemalloc

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _dense(q, k, v, causal=False):
    s = (q @ k.T) / math.sqrt(q.shape[-1])
    if causal:
        s = np.where(np.tril(np.ones(s.shape, dtype=bool)), s, -np.inf)
    return _softmax(s) @ v


def _data(T, d=4, dv=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, dv))


@check("online_stats finds the running max and sum from a one-shot stream (worked: m = 5, l = 1.2034)")
def test_online_stats(sub):
    fn = has(sub, "online_stats")
    m, l = fn(iter([2.0, 1.0, 5.0, 3.0]))
    almost(m, 5.0, 1e-12, label="running max m")
    almost(l, 1.2034, 1e-4, label="running sum l")
    x = np.random.default_rng(0).normal(size=50) * 30
    m, l = fn(float(t) for t in x)
    almost(m, x.max(), 1e-12, label="running max on 50 random scores")
    almost(l, np.exp(x - x.max()).sum(), 1e-10, label="running sum on 50 random scores")


@check("online_softmax equals softmax to 1e-12, with the max arriving late and with huge scores")
def test_online_softmax(sub):
    fn = has(sub, "online_softmax")
    close(fn(np.array([2.0, 1.0, 5.0, 3.0])), np.array([0.0414, 0.0152, 0.8310, 0.1125]), atol=1e-4, label="worked example")
    rng = np.random.default_rng(1)
    for x in (rng.normal(size=20), np.sort(rng.normal(size=20)), 1000.0 + rng.normal(size=20)):
        got = fn(x)
        close(got, _softmax(x), atol=1e-12, rtol=0, label="online_softmax")
        close(got.sum(), 1.0, atol=1e-12, label="sum of the probabilities")


@check("merge_states of two blocks matches the dense row, and the merge is associative")
def test_merge(sub):
    merge = has(sub, "merge_states")
    rng = np.random.default_rng(2)
    n, dv = 3, 2
    s = 3.0 * rng.normal(size=(n, 9))
    v = rng.normal(size=(9, dv))

    def state(lo, hi):
        m = s[:, lo:hi].max(axis=-1)
        e = np.exp(s[:, lo:hi] - m[:, None])
        return m, e.sum(axis=-1), e @ v[lo:hi]

    a, b, c = state(0, 2), state(2, 6), state(6, 9)
    left = merge(merge(a, b), c)
    right = merge(a, merge(b, c))
    for name, x, y in zip(("m", "l", "o"), left, right):
        close(x, y, atol=1e-12, label=f"associativity of {name}")
    m, l, o = left
    close(o / l[:, None], _softmax(s) @ v, atol=1e-12, label="merged output against the dense row")
    close(m + np.log(l), np.log(np.exp(s).sum(axis=-1)), atol=1e-10, label="merged log-sum-exp")


@check("tiled_attention equals dense attention for several block sizes")
def test_tiled(sub):
    fn = has(sub, "tiled_attention")
    q, k, v = _data(13)
    want = _dense(q, k, v)
    for block in (1, 3, 4, 13, 20):
        out, lse, peak = fn(q, k, v, block)
        shape(out, (13, 3), label=f"out, block {block}")
        close(out, want, atol=1e-12, label=f"out, block {block}")
        s = (q @ k.T) / 2.0
        close(lse, np.log(np.exp(s).sum(axis=-1)), atol=1e-10, label=f"lse, block {block}")


# Catches a tiled attention that forgets to rescale the running output when the running
# max changes. It adds exp(s - m_new) @ v into o but leaves the old o on the old max. That
# is right when the largest score sits in the first key block and wrong otherwise, so this
# check puts one huge score in the last block. On the lesson's example it returns 43.85
# instead of 30.145.
@check("tiled_attention rescales the running output when the max arrives in a late block")
def test_late_max(sub):
    fn = has(sub, "tiled_attention")
    q, k, v = _data(12, seed=5)
    k = k.copy()
    k[10] = 9.0 * q[0] / np.linalg.norm(q[0])
    q = np.repeat(q[:1], 12, axis=0)
    want = _dense(q, k, v)
    for block in (2, 4, 6):
        out, _, _ = fn(q, k, v, block)
        close(out, want, atol=1e-10, label=f"out with a late max, block {block}")
    # The worked example: d = 1 and q = 1, so the scores are the keys [2, 1, 5, 3].
    qq = np.array([[1.0]])
    kk = np.array([[2.0], [1.0], [5.0], [3.0]])
    vv = np.array([[10.0], [20.0], [30.0], [40.0]])
    out, _, _ = fn(qq, kk, vv, 2)
    close(out, np.array([[30.145]]), atol=1e-3, label="worked example out")


@check("tiled_attention with the causal mask equals dense, and holds only block x block scores")
def test_causal_peak(sub):
    fn = has(sub, "tiled_attention")
    q, k, v = _data(14, seed=3)
    want = _dense(q, k, v, causal=True)
    for block in (1, 2, 4, 5, 14):
        out, lse, peak = fn(q, k, v, block, causal=True)
        close(out, want, atol=1e-12, label=f"causal out, block {block}")
        equal(peak, block * block, label=f"peak score entries, block {block}")
    q, k, v = _data(1024, d=8, seed=4)
    tracemalloc.start()
    try:
        fn(q, k, v, 32)
        _, top = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if top > 2_000_000:
        fail(
            f"peak extra memory was {top:,} bytes for T = 1024, block = 32; expected under "
            f"2,000,000, while the dense table alone is {1024 * 1024 * 8:,} bytes"
        )


@check("count_causal_blocks counts the tiles on or below the diagonal")
def test_blocks(sub):
    fn = has(sub, "count_causal_blocks")
    equal(tuple(fn(8, 2)), (10, 6), label="T = 8, block = 2")
    equal(tuple(fn(4096, 128)), (528, 496), label="T = 4096, block = 128")
    equal(tuple(fn(10, 4)), (6, 3), label="T = 10, block = 4 (last tile is short)")
    equal(tuple(fn(5, 5)), (1, 0), label="one tile")


@check("recompute_probs rebuilds the weights exactly from lse, with and without the mask")
def test_recompute(sub):
    tiled = has(sub, "tiled_attention")
    fn = has(sub, "recompute_probs")
    q, k, v = _data(9, seed=6)
    for causal in (False, True):
        _, lse, _ = tiled(q, k, v, 4, causal=causal)
        s = (q @ k.T) / 2.0
        if causal:
            s = np.where(np.tril(np.ones(s.shape, dtype=bool)), s, -np.inf)
        p = fn(q, k, lse, causal=causal)
        close(p, _softmax(s), atol=1e-12, rtol=0, label=f"weights, causal={causal}")
        close(p.sum(axis=-1), np.ones(9), atol=1e-12, label=f"row sums, causal={causal}")


@check("backward_from_lse matches finite differences for dq, dk, and dv, with and without the mask")
def test_backward(sub):
    tiled = has(sub, "tiled_attention")
    back = has(sub, "backward_from_lse")
    for causal in (False, True):
        q, k, v = _data(5, seed=7)
        dout = 0.3 * np.random.default_rng(8).normal(size=(5, 3))
        out, lse, _ = tiled(q, k, v, 2, causal=causal)
        dq, dk, dv = back(dout, q, k, v, out, lse, causal=causal)

        def loss(a, b, c):
            return float(np.sum(_dense(a, b, c, causal) * dout))

        close(dq, finite_difference(lambda a: loss(a, k, v), q.copy()), atol=1e-6, rtol=1e-5, label=f"dq, causal={causal}")
        close(dk, finite_difference(lambda a: loss(q, a, v), k.copy()), atol=1e-6, rtol=1e-5, label=f"dk, causal={causal}")
        close(dv, finite_difference(lambda a: loss(q, k, a), v.copy()), atol=1e-6, rtol=1e-5, label=f"dv, causal={causal}")


@check("extra_saved and score_table_flops give the memory and flop bills")
def test_counts(sub):
    saved = has(sub, "extra_saved")
    flops = has(sub, "score_table_flops")
    equal(saved(4096, False), 2 * 4096 * 4096, label="dense saves S and P")
    equal(saved(4096, True), 4096, label="tiled saves lse")
    equal(flops(4096, 1024, "forward"), 68_719_476_736, label="forward flops, 4 T^2 d")
    equal(flops(4096, 1024, "backward"), 137_438_953_472, label="backward flops, 8 T^2 d")
    equal(flops(4096, 1024, "backward_recompute"), 171_798_691_840, label="backward with recompute, 10 T^2 d")
