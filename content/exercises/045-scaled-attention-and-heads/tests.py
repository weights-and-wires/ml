"""Checks for the scaled-attention, causal-mask, and multi-head exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _manual_mha(x, Wq, Wk, Wv, Wo, heads, mask=None):
    """Reference written inside the test, one head at a time."""
    T, d = x.shape
    d_k = d // heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    outs = []
    for i in range(heads):
        sl = slice(i * d_k, (i + 1) * d_k)
        s = Q[:, sl] @ K[:, sl].T / np.sqrt(d_k)
        if mask is not None:
            s = s + mask
        outs.append(_softmax(s) @ V[:, sl])
    return np.concatenate(outs, axis=1) @ Wo


def _weights(rng, d):
    return [0.4 * rng.normal(size=(d, d)) for _ in range(4)]


@check("dot_scores gives the table of query-key dot products, with no scaling")
def test_dot_scores(sub):
    fn = has(sub, "dot_scores")
    q = np.array([[1.0, 1.0]])
    k = np.array([[1.5, 0.5], [0.5, 0.5]])
    got = fn(q, k)
    shape(got, (1, 2), label="dot_scores output")
    close(got, np.array([[2.0, 1.0]]), atol=1e-12, label="scores of the worked example")
    q64 = np.array([[0.25] * 64, [0.125] * 64])
    k64 = np.array([[1.0] * 64, [0.5] * 64])
    close(fn(q64, k64), np.array([[16.0, 8.0], [8.0, 4.0]]), atol=1e-9,
          label="unscaled scores for 64-wide vectors")
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 5))
    k = rng.normal(size=(4, 5))
    got = fn(q, k)
    shape(got, (3, 4), label="dot_scores output")
    for i in range(3):
        for j in range(4):
            want = float(np.dot(q[i], k[j]))
            if abs(got[i, j] - want) > 1e-10:
                fail(f"entry [{i}, {j}] is {got[i, j]:.6f}, expected {want:.6f}")


@check("softmax gives [0.7311, 0.2689] for [2, 1], stays finite at 900, and shifts each row by its own maximum")
def test_softmax(sub):
    fn = has(sub, "softmax")
    close(fn(np.array([2.0, 1.0])), np.array([0.7310585786, 0.2689414214]), atol=1e-8,
          label="softmax([2, 1])")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4)) * 5
    rows = fn(x, axis=-1)
    shape(rows, (3, 4), label="softmax output")
    close(rows.sum(axis=1), np.ones(3), atol=1e-10, label="row sums")
    close(fn(x, axis=0).sum(axis=0), np.ones(4), atol=1e-10, label="column sums")

    out = fn(np.array([900.0, 0.0]))
    if not np.all(np.isfinite(out)):
        fail(f"softmax([900, 0]) returned non-finite values {out}")
    close(out, np.array([1.0, 0.0]), atol=1e-12, label="softmax([900, 0])")
    z = np.array([0.3, -1.2, 2.0, 0.0])
    close(fn(z + 1000.0), fn(z), atol=1e-10, label="softmax after adding 1000 to every score")
    x = np.array([[900.0, 0.0], [0.0, -900.0], [-1000.0, -1001.0]])
    out = fn(x, axis=-1)
    if not np.all(np.isfinite(out)):
        fail(f"softmax returned non-finite values for rows of different size: {out}; "
             "subtract the maximum of each row, not one maximum for the whole array")
    close(out[2], np.array([0.7310585786, 0.2689414214]), atol=1e-8, label="row [-1000, -1001]")
    close(out[:2], np.array([[1.0, 0.0], [1.0, 0.0]]), atol=1e-12, label="first two rows")


@check("blend_values returns a row of v for one-hot weights and the average for uniform weights")
def test_blend(sub):
    fn = has(sub, "blend_values")
    rng = np.random.default_rng(2)
    v = rng.normal(size=(4, 3))
    weights = np.zeros((2, 4))
    weights[0, 2] = 1.0
    weights[1, 0] = 1.0
    got = fn(weights, v)
    shape(got, (2, 3), label="blend_values output")
    close(got[0], v[2], atol=1e-12, label="row picked by a one-hot weight")
    close(got[1], v[0], atol=1e-12, label="second row picked")
    v = rng.normal(size=(5, 2))
    close(fn(np.full((1, 5), 0.2), v), v.mean(axis=0, keepdims=True), atol=1e-12,
          label="uniform blend")


@check("attention_score_scale is 1 / sqrt(d_k)")
def test_scale(sub):
    fn = has(sub, "attention_score_scale")
    for d_k, want in [(1, 1.0), (4, 0.5), (16, 0.25), (64, 0.125), (256, 0.0625)]:
        got = fn(d_k)
        if abs(float(got) - want) > 1e-12:
            fail(f"attention_score_scale({d_k}) is {got}, expected {want}")


@check("score_std is near sqrt(d_k) for random vectors, and near 1 after scaling")
def test_score_std(sub):
    fn = has(sub, "score_std")
    scale = has(sub, "attention_score_scale")
    for d_k in (4, 16, 64, 256):
        got = fn(d_k)
        want = np.sqrt(d_k)
        if abs(got - want) > 0.10 * want:
            fail(f"score_std({d_k}) is {got:.3f}, expected about {want:.3f} (within 10 percent)")
        scaled = fn(d_k, n=2000, seed=1) * scale(d_k)
        if abs(scaled - 1.0) > 0.10:
            fail(f"spread after scaling at d_k={d_k} is {scaled:.3f}, expected about 1")
    if fn(16, seed=5) != fn(16, seed=5):
        fail("score_std must be repeatable for the same seed")


@check("scaled_dot_product_attention gives the right shapes, the worked 64-wide weights, the formula, and no one-hot rows at width 256")
def test_attention(sub):
    fn = has(sub, "scaled_dot_product_attention")
    rng = np.random.default_rng(0)
    Tq, Tk, d_k, d_v = 4, 6, 5, 3
    q = rng.normal(size=(Tq, d_k))
    k = rng.normal(size=(Tk, d_k))
    v = rng.normal(size=(Tk, d_v))
    out, weights = fn(q, k, v)
    shape(out, (Tq, d_v), label="attention output")
    shape(weights, (Tq, Tk), label="attention weights")
    if np.any(weights < 0):
        fail("attention weights must not be negative")
    close(weights.sum(axis=-1), np.ones(Tq), atol=1e-10, label="row sums of the weights")

    q = np.array([[0.25] * 64, [0.125] * 64])
    k = np.array([[1.0] * 64, [0.5] * 64])
    out, weights = fn(q, k, np.eye(2))
    want = np.array([[0.7310585786, 0.2689414214], [0.6224593312, 0.3775406688]])
    close(weights, want, atol=1e-8, label="weights of the 64-wide example")
    close(out, want, atol=1e-8, label="output with identity values")

    rng = np.random.default_rng(7)
    q = rng.normal(size=(5, 9))
    k = rng.normal(size=(5, 9))
    v = rng.normal(size=(5, 4))
    s = q @ k.T / 3.0
    s = s - s.max(axis=1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(axis=1, keepdims=True)
    out, weights = fn(q, k, v)
    close(weights, w, atol=1e-10, label="weights against softmax(q k^T / sqrt(d_k))")
    close(out, w @ v, atol=1e-10, label="output against softmax(q k^T / sqrt(d_k)) v")

    rng = np.random.default_rng(3)
    q = rng.normal(size=(40, 256))
    k = rng.normal(size=(8, 256))
    _, weights = fn(q, k, np.eye(8))
    mean_max = float(weights.max(axis=1).mean())
    if mean_max > 0.5:
        fail(f"the average largest weight is {mean_max:.3f}; scaled scores at d_k=256 should give about 0.36")


@check("causal_mask is 0 on and below the diagonal and -inf above, and gives exact zeros after a softmax")
def test_causal_mask(sub):
    fn = has(sub, "causal_mask")
    T = 5
    mask = fn(T)
    shape(mask, (T, T), label="causal_mask output")
    lower = np.tril(np.ones((T, T), dtype=bool))
    if not np.all(mask[lower] == 0.0):
        fail("every entry on or below the diagonal must be 0, including the diagonal itself")
    if not np.all(np.isneginf(mask[~lower])):
        fail("every entry strictly above the diagonal must be -inf")
    close(fn(1), np.zeros((1, 1)), atol=0, label="mask for one token")

    rng = np.random.default_rng(0)
    w = _softmax(rng.normal(size=(5, 5)) + mask)
    if not np.all(np.isfinite(w)):
        fail("masked softmax produced non-finite values; check that the diagonal is 0, not -inf")
    if np.any(np.triu(w, k=1) != 0.0):
        fail("weights above the diagonal must be exactly 0")
    close(w.sum(axis=1), np.ones(5), atol=1e-12, label="row sums of the masked weights")
    close(w[0], np.array([1.0, 0, 0, 0, 0]), atol=1e-12, label="row 0 can only read itself")


@check("split_heads returns column blocks, not row blocks")
def test_split_heads(sub):
    fn = has(sub, "split_heads")
    x = np.arange(24, dtype=float).reshape(3, 8)
    parts = fn(x, 4)
    if len(parts) != 4:
        fail(f"expected 4 heads, got {len(parts)}")
    for i, part in enumerate(parts):
        shape(part, (3, 2), label=f"head {i}")
        close(part, x[:, 2 * i:2 * i + 2], atol=0, label=f"head {i} contents (columns {2*i} and {2*i+1})")
    close(np.concatenate(parts, axis=1), x, atol=0, label="heads joined back together")
    try:
        fn(x, 3)
    except ValueError:
        return
    fail("split_heads(x, 3) with d = 8 should raise ValueError")


@check("multi_head_attention matches single-head attention for one head, the worked example, and a head-by-head loop")
def test_multi_head(sub):
    mha = has(sub, "multi_head_attention")
    sdpa = has(sub, "scaled_dot_product_attention")
    causal = has(sub, "causal_mask")
    rng = np.random.default_rng(1)
    T, d = 5, 8
    x = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(rng, d)
    got = mha(x, Wq, Wk, Wv, Wo, heads=1)
    want = sdpa(x @ Wq, x @ Wk, x @ Wv)[0] @ Wo
    close(got, want, atol=1e-9, label="one-head output")

    x2 = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    eye = np.eye(4)
    got = mha(x2, eye, eye, eye, eye, heads=2)
    want = np.array([[0.66976155, 0.33023845, 1.0, 1.0], [0.33023845, 0.66976155, 1.0, 1.0]])
    close(got, want, atol=1e-6, label="output of the two-head example")

    rng = np.random.default_rng(2)
    T, d = 6, 8
    x = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(rng, d)
    for heads in (2, 4):
        close(mha(x, Wq, Wk, Wv, Wo, heads=heads), _manual_mha(x, Wq, Wk, Wv, Wo, heads),
              atol=1e-9, label=f"{heads} heads, no mask")
        mask = causal(T)
        close(mha(x, Wq, Wk, Wv, Wo, heads=heads, mask=mask),
              _manual_mha(x, Wq, Wk, Wv, Wo, heads, mask),
              atol=1e-9, label=f"{heads} heads, causal mask")


@check("with the causal mask later tokens never change earlier outputs, and without it a shuffle of the input shuffles the output")
def test_mask_and_order(sub):
    mha = has(sub, "multi_head_attention")
    causal = has(sub, "causal_mask")
    rng = np.random.default_rng(3)
    T, d = 6, 8
    x1 = rng.normal(size=(T, d))
    x2 = x1.copy()
    x2[3:] = 5.0 * rng.normal(size=(T - 3, d))
    Wq, Wk, Wv, Wo = _weights(rng, d)
    mask = causal(T)
    for heads in (2, 4):
        o1 = mha(x1, Wq, Wk, Wv, Wo, heads=heads, mask=mask)
        o2 = mha(x2, Wq, Wk, Wv, Wo, heads=heads, mask=mask)
        close(o1[:3], o2[:3], atol=1e-10, label=f"outputs 0-2 with {heads} heads")
        if np.max(np.abs(o1[3:] - o2[3:])) < 1e-6:
            fail("the later outputs should change when the later inputs change; the setup is broken")

    rng = np.random.default_rng(4)
    x = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(rng, d)
    perm = rng.permutation(T)
    out = mha(x, Wq, Wk, Wv, Wo, heads=2)
    out_perm = mha(x[perm], Wq, Wk, Wv, Wo, heads=2)
    close(out_perm, out[perm], atol=1e-9, label="output of the shuffled input")
