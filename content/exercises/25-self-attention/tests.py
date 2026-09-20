"""Checks for the self-attention exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


@check("causal_mask is zero on and below the diagonal and -inf above it")
def test_causal_mask_structure(sub):
    fn = has(sub, "causal_mask")
    T = 5
    mask = fn(T)
    shape(mask, (T, T), label="causal_mask output")

    lower = np.tril(np.ones((T, T), dtype=bool))
    if not np.all(mask[lower] == 0.0):
        fail("every entry on or below the diagonal of the causal mask must be 0")
    upper = ~lower
    if not np.all(np.isneginf(mask[upper])):
        fail("every entry strictly above the diagonal of the causal mask must be -inf")


@check("softmax is stable at large logits and sums to one along the given axis")
def test_softmax_stable(sub):
    fn = has(sub, "softmax")
    z = np.array([-900.0, 0.0, 900.0])
    out = fn(z)
    if not np.all(np.isfinite(out)):
        fail(f"softmax returned non-finite values at large logits: {out}")
    close(out, np.array([0.0, 0.0, 1.0]), atol=1e-8, label="softmax at large logits")

    batch = np.array([[1.0, 2.0, 3.0], [800.0, 0.0, -800.0]])
    out_rows = fn(batch, axis=-1)
    close(out_rows.sum(axis=-1), np.ones(2), atol=1e-10, label="row sums")

    out_cols = fn(batch, axis=0)
    close(out_cols.sum(axis=0), np.ones(3), atol=1e-10, label="column sums")


@check("scaled_dot_product_attention returns a valid distribution and the right shape")
def test_sdpa_basic(sub):
    rng = np.random.default_rng(0)
    Tq, Tk, d_k, d_v = 4, 6, 5, 3
    q = rng.normal(size=(Tq, d_k))
    k = rng.normal(size=(Tk, d_k))
    v = rng.normal(size=(Tk, d_v))

    fn = has(sub, "scaled_dot_product_attention")
    out, weights = fn(q, k, v)
    shape(out, (Tq, d_v), label="attention output")
    shape(weights, (Tq, Tk), label="attention weights")
    if np.any(weights < -1e-10):
        fail(f"attention weights must be non-negative, got minimum {weights.min():.3g}")
    close(weights.sum(axis=-1), np.ones(Tq), atol=1e-8, label="row sums of the weights")


@check("the causal mask makes a position's output independent of anything after it")
def test_causal_independence(sub):
    sdpa = has(sub, "scaled_dot_product_attention")
    causal = has(sub, "causal_mask")

    rng = np.random.default_rng(1)
    T, d_k = 6, 4
    x1 = rng.normal(size=(T, d_k))
    x2 = x1.copy()
    x2[3:] = rng.normal(size=(T - 3, d_k)) * 5.0  # perturb only positions 3, 4, 5

    mask = causal(T)
    out1, _ = sdpa(x1, x1, x1, mask)
    out2, _ = sdpa(x2, x2, x2, mask)

    close(out1[:3], out2[:3], atol=1e-10,
          label="output at positions 0-2 after perturbing positions 3-5")

    if np.max(np.abs(out1[3:] - out2[3:])) < 1e-6:
        fail("perturbing positions 3-5 should change their own output; the test setup is broken")


@check("multi-head attention with one head equals single-head attention exactly")
def test_single_head_matches(sub):
    rng = np.random.default_rng(2)
    T, d = 6, 8
    x = rng.normal(size=(T, d))
    Wq = 0.3 * rng.normal(size=(d, d))
    Wk = 0.3 * rng.normal(size=(d, d))
    Wv = 0.3 * rng.normal(size=(d, d))
    Wo = 0.3 * rng.normal(size=(d, d))

    mha = has(sub, "multi_head_attention")
    sdpa = has(sub, "scaled_dot_product_attention")
    causal = has(sub, "causal_mask")
    mask = causal(T)

    got = mha(x, Wq, Wk, Wv, Wo, heads=1, mask=mask)
    single_out, _ = sdpa(x @ Wq, x @ Wk, x @ Wv, mask)
    want = single_out @ Wo
    close(got, want, atol=1e-8, label="one-head multi-head attention")


@check("permuting the rows of the input permutes the rows of the output identically")
def test_permutation_equivariance(sub):
    rng = np.random.default_rng(3)
    T, d, heads = 6, 8, 2
    x = rng.normal(size=(T, d))
    Wq = 0.3 * rng.normal(size=(d, d))
    Wk = 0.3 * rng.normal(size=(d, d))
    Wv = 0.3 * rng.normal(size=(d, d))
    Wo = 0.3 * rng.normal(size=(d, d))

    mha = has(sub, "multi_head_attention")
    out = mha(x, Wq, Wk, Wv, Wo, heads)  # no mask: masking is not permutation equivariant

    perm = rng.permutation(T)
    out_perm = mha(x[perm], Wq, Wk, Wv, Wo, heads)
    close(out_perm, out[perm], atol=1e-8, label="output of the permuted input")


@check("attention_backward matches finite differences for q, k, and v")
def test_attention_backward(sub):
    rng = np.random.default_rng(4)
    T, d_k, d_v = 5, 4, 3
    q = rng.normal(size=(T, d_k))
    k = rng.normal(size=(T, d_k))
    v = rng.normal(size=(T, d_v))
    causal = has(sub, "causal_mask")
    mask = causal(T)

    sdpa = has(sub, "scaled_dot_product_attention")
    backward = has(sub, "attention_backward")

    dout = 0.2 * rng.normal(size=(T, d_v))
    dq, dk, dv = backward(dout, q, k, v, mask)

    def loss_q(arr):
        out, _ = sdpa(arr, k, v, mask)
        return float(np.sum(out * dout))

    def loss_k(arr):
        out, _ = sdpa(q, arr, v, mask)
        return float(np.sum(out * dout))

    def loss_v(arr):
        out, _ = sdpa(q, k, arr, mask)
        return float(np.sum(out * dout))

    close(dq, finite_difference(loss_q, q), atol=1e-4, rtol=1e-3, label="dq")
    close(dk, finite_difference(loss_k, k), atol=1e-4, rtol=1e-3, label="dk")
    close(dv, finite_difference(loss_v, v), atol=1e-4, rtol=1e-3, label="dv")


@check("attention_flops matches the hand-computed value and does not depend on the head count")
def test_attention_flops(sub):
    fn = has(sub, "attention_flops")
    for T, d in [(16, 32), (64, 16), (128, 64)]:
        want = 8 * T * d ** 2 + 4 * T ** 2 * d
        values = {heads: fn(T, d, heads) for heads in (1, 2, 4, 8) if d % heads == 0}
        for heads, got in values.items():
            if got != want:
                fail(
                    f"attention_flops({T}, {d}, {heads}) is {got}, expected {want} "
                    "from 8*T*d**2 + 4*T**2*d"
                )
        if len(set(values.values())) != 1:
            fail(f"attention_flops should not depend on the head count, got {values}")
