"""Checks for the attention backward and cost exercise. The forward functions are provided."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@check("softmax_backward reproduces the worked example [0.3932, -0.3932]")
def test_softmax_backward_hand(sub):
    fn = has(sub, "softmax_backward")
    w = _softmax(np.array([[2.0, 1.0]]))
    got = fn(w, np.array([[1.0, -1.0]]))
    shape(got, (1, 2), label="softmax_backward output")
    close(got, np.array([[0.3932, -0.3932]]), atol=1e-4, label="gradient of the scores")


@check("softmax_backward rows sum to 0 and match finite differences")
def test_softmax_backward_fd(sub):
    fn = has(sub, "softmax_backward")
    rng = np.random.default_rng(0)
    s = rng.normal(size=(3, 5))
    dw = rng.normal(size=(3, 5))
    got = fn(_softmax(s), dw)
    close(got.sum(axis=-1), np.zeros(3), atol=1e-10, label="row sums of the score gradient")
    want = finite_difference(lambda arr: float(np.sum(_softmax(arr) * dw)), s)
    close(got, want, atol=1e-6, rtol=1e-5, label="score gradient")


@check("attention_backward reproduces the worked example for dq, dk, and dv")
def test_backward_hand(sub):
    fn = has(sub, "attention_backward")
    q = np.ones((1, 4))
    k = np.array([[1.0] * 4, [0.5] * 4])
    v = np.eye(2)
    dq, dk, dv = fn(np.array([[1.0, -1.0]]), q, k, v)
    shape(dq, (1, 4), label="dq")
    shape(dk, (2, 4), label="dk")
    shape(dv, (2, 2), label="dv")
    close(dq, np.full((1, 4), 0.0983), atol=1e-4, label="dq")
    close(dk, np.array([[0.1966] * 4, [-0.1966] * 4]), atol=1e-4, label="dk")
    close(dv, np.array([[0.7311, -0.7311], [0.2689, -0.2689]]), atol=1e-4, label="dv")


def _fd_check(sub, mask):
    sdpa = has(sub, "scaled_dot_product_attention")
    backward = has(sub, "attention_backward")
    rng = np.random.default_rng(4)
    T, d_k, d_v = 5, 4, 3
    q = rng.normal(size=(T, d_k))
    k = rng.normal(size=(T, d_k))
    v = rng.normal(size=(T, d_v))
    dout = 0.3 * rng.normal(size=(T, d_v))
    dq, dk, dv = backward(dout, q, k, v, mask)

    def loss(qq, kk, vv):
        return float(np.sum(sdpa(qq, kk, vv, mask)[0] * dout))

    close(dq, finite_difference(lambda a: loss(a, k, v), q.copy()), atol=1e-5, rtol=1e-4, label="dq")
    close(dk, finite_difference(lambda a: loss(q, a, v), k.copy()), atol=1e-5, rtol=1e-4, label="dk")
    close(dv, finite_difference(lambda a: loss(q, k, a), v.copy()), atol=1e-5, rtol=1e-4, label="dv")


@check("attention_backward matches finite differences for dq, dk, and dv")
def test_backward_fd(sub):
    _fd_check(sub, None)


@check("attention_backward matches finite differences under a causal mask")
def test_backward_masked(sub):
    T = 5
    mask = np.zeros((T, T))
    mask[np.triu_indices(T, k=1)] = -np.inf
    _fd_check(sub, mask)


@check("attention_backward keeps the 1/sqrt(d_k) factor: dq shrinks as the key width grows")
def test_backward_scale(sub):
    backward = has(sub, "attention_backward")
    rng = np.random.default_rng(6)
    T = 4
    k1 = rng.normal(size=(T, 4))
    q1 = rng.normal(size=(T, 4))
    v = rng.normal(size=(T, 3))
    dout = rng.normal(size=(T, 3))
    # Repeat every column 4 times (width 16) and halve q, so the raw scores double and the
    # scaled scores, which divide by sqrt(16) = 4 instead of sqrt(4) = 2, are unchanged.
    # The forward pass is then identical, so dscores is too, and only the factor
    # 1/sqrt(d_k) differs: each column of dq at width 16 is half of dq at width 4.
    q2 = 0.5 * np.repeat(q1, 4, axis=1)
    k2 = np.repeat(k1, 4, axis=1)
    dq1 = backward(dout, q1, k1, v)[0]
    dq2 = backward(dout, q2, k2, v)[0]
    close(dq2[:, ::4], dq1 / 2.0, atol=1e-9, label="dq at width 16 against width 4")


@check("attention_flops is 8Td^2 + 4T^2 d, an int, and ignores the head count")
def test_flops(sub):
    fn = has(sub, "attention_flops")
    got = fn(16, 4, 1)
    if got != 6144:
        fail(f"attention_flops(16, 4, 1) is {got}, expected 6144 = 8*16*16 + 4*256*4")
    for T, d in [(16, 32), (64, 16), (128, 64)]:
        want = 8 * T * d ** 2 + 4 * T ** 2 * d
        for heads in (1, 2, 4, 8):
            val = fn(T, d, heads)
            if val != want or not isinstance(val, (int, np.integer)):
                fail(f"attention_flops({T}, {d}, {heads}) is {val!r}, expected the int {want}")


@check("the score table overtakes the projections once T is larger than 2d")
def test_crossover(sub):
    fn = has(sub, "attention_flops")
    d = 64
    proj = lambda T: 8 * T * d * d
    # Below the crossover the projections dominate, above it the T^2 term does.
    if not fn(64, d, 1) - proj(64) < proj(64):
        fail("at T = 64 = d the score table should cost less than the projections")
    if not fn(256, d, 1) - proj(256) > proj(256):
        fail("at T = 256 = 4d the score table should cost more than the projections")
    ratio = fn(8192, d, 1) / fn(4096, d, 1)
    if not 3.8 < ratio < 4.0:
        fail(
            f"doubling a long sequence should nearly quadruple the cost, "
            f"got a factor {ratio:.3f}, expected a factor between 3.8 and 4"
        )


def _mha_forward(x, Wq, Wk, Wv, Wo, heads, mask):
    """Multi-head forward used as the finite-difference loss. Not the submission."""
    d_h = x.shape[-1] // heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    parts = []
    for i in range(heads):
        sl = slice(i * d_h, (i + 1) * d_h)
        scale = 1.0 / np.sqrt(d_h)
        scores = (Q[:, sl] @ K[:, sl].T) * scale
        if mask is not None:
            scores = scores + mask
        parts.append(_softmax(scores) @ V[:, sl])
    return np.concatenate(parts, axis=-1) @ Wo


def _reference_mha_backward(dout, x, Wq, Wk, Wv, Wo, heads, mask):
    """Head-by-head backward written out in the check.

    torch is not a dependency of this exercise, so this reference stands in
    for an autograd comparison. It does not call the submission.
    """
    d_h = x.shape[-1] // heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    outputs = []
    saved = []
    for i in range(heads):
        sl = slice(i * d_h, (i + 1) * d_h)
        scale = 1.0 / np.sqrt(d_h)
        scores = (Q[:, sl] @ K[:, sl].T) * scale
        if mask is not None:
            scores = scores + mask
        weights = _softmax(scores)
        saved.append(weights)
        outputs.append(weights @ V[:, sl])
    concat = np.concatenate(outputs, axis=-1)
    d_concat = dout @ Wo.T
    dWo = concat.T @ dout
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)
    for i in range(heads):
        sl = slice(i * d_h, (i + 1) * d_h)
        weights = saved[i]
        upstream = d_concat[:, sl]
        dv = weights.T @ upstream
        dweights = upstream @ V[:, sl].T
        rowsum = np.sum(weights * dweights, axis=-1, keepdims=True)
        dscores = weights * (dweights - rowsum)
        scale = 1.0 / np.sqrt(d_h)
        dQ[:, sl] = (dscores @ K[:, sl]) * scale
        dK[:, sl] = (dscores.T @ Q[:, sl]) * scale
        dV[:, sl] = dv
    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    return dx, x.T @ dQ, x.T @ dK, x.T @ dV, dWo


def _causal_mask(T):
    mask = np.zeros((T, T))
    mask[np.triu_indices(T, k=1)] = -np.inf
    return mask


# Catches a head concatenation that forgets to split the upstream gradient.
# d_concat = dout @ Wo.T has shape (T, d), and each head must receive only its
# own d_h columns. Passing the full d_concat into one unsplit attention on Q,
# K, and V keeps every shape and still applies the causal mask, but on the
# worked example the value gradient comes out
# [[0.3302, -0.3302], [0.6698, -0.6698]] instead of
# [[0.5, -0.2689], [0.5, -0.7311]].
@check("mha_backward reproduces the two-head causal example")
def test_mha_hand(sub):
    fn = has(sub, "mha_backward")
    eye = np.eye(2)
    got = fn(
        np.array([[0.0, 0.0], [1.0, -1.0]]),
        eye, eye, eye, eye, eye, 2, _causal_mask(2),
    )
    if not isinstance(got, tuple) or len(got) != 5:
        fail(
            f"mha_backward returned {type(got).__name__} "
            f"of length {len(got) if isinstance(got, tuple) else 'n/a'}, "
            "expected a tuple of 5 arrays (dx, dWq, dWk, dWv, dWo)"
        )
    dx, dWq, dWk, dWv, dWo = got
    shape(dx, (2, 2), label="dx")
    shape(dWq, (2, 2), label="dWq")
    shape(dWk, (2, 2), label="dWk")
    shape(dWv, (2, 2), label="dWv")
    shape(dWo, (2, 2), label="dWo")
    close(dWv, np.array([[0.5, -0.2689], [0.5, -0.7311]]), atol=1e-4, label="dV")
    close(dWo, np.array([[0.5, -0.5], [0.7311, -0.7311]]), atol=1e-4, label="dWo")
    close(dx, np.array([[0.5, -0.0723], [0.75, -1.1243]]), atol=1e-4, label="dx")
    close(dWq, np.array([[0.0, 0.0], [0.25, -0.1966]]), atol=1e-4, label="dWq")
    close(dWk, np.array([[0.0, 0.1966], [0.0, -0.1966]]), atol=1e-4, label="dWk")


@check("mha_backward matches a head-by-head reference and finite differences under a causal mask")
def test_mha_reference_and_fd(sub):
    fn = has(sub, "mha_backward")
    rng = np.random.default_rng(1)
    T, d, heads = 3, 4, 2
    x = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) for _ in range(4))
    dout = rng.normal(size=(T, d))
    mask = _causal_mask(T)
    got = fn(dout, x, Wq, Wk, Wv, Wo, heads, mask)
    if not isinstance(got, tuple) or len(got) != 5:
        fail(
            f"mha_backward returned {type(got).__name__} "
            f"of length {len(got) if isinstance(got, tuple) else 'n/a'}, "
            "expected a tuple of 5 arrays (dx, dWq, dWk, dWv, dWo)"
        )
    want = _reference_mha_backward(dout, x, Wq, Wk, Wv, Wo, heads, mask)
    labels = ("dx", "dWq", "dWk", "dWv", "dWo")
    shapes = ((T, d), (d, d), (d, d), (d, d), (d, d))
    for label, expect_shape, pred, truth in zip(labels, shapes, got, want):
        shape(pred, expect_shape, label=label)
        close(pred, truth, atol=1e-5, rtol=1e-4, label=f"{label} against the head-by-head reference")

    arrays = (x, Wq, Wk, Wv, Wo)

    def loss_at(index, value):
        args = [arr.copy() for arr in arrays]
        args[index] = value
        return float(np.sum(_mha_forward(args[0], *args[1:], heads, mask) * dout))

    for index, label in enumerate(labels):
        fd = finite_difference(lambda value, index=index: loss_at(index, value), arrays[index].copy())
        close(got[index], fd, atol=1e-5, rtol=1e-4, label=f"{label} against finite differences")
