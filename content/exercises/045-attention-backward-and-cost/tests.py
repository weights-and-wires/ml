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
        fail(f"doubling a long sequence should nearly quadruple the cost, got a factor {ratio:.3f}")
