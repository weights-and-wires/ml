"""Checks for the heads and causal mask exercise. The softmax and single-head attention are provided."""

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


@check("causal_mask is 0 on and below the diagonal and -inf strictly above it")
def test_mask_structure(sub):
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


@check("adding the mask before the softmax gives exact zeros above the diagonal and rows that sum to 1")
def test_mask_in_softmax(sub):
    fn = has(sub, "causal_mask")
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(5, 5))
    w = _softmax(scores + fn(5))
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


@check("multi-head attention with one head equals single-head attention followed by Wo")
def test_one_head(sub):
    mha = has(sub, "multi_head_attention")
    sdpa = has(sub, "scaled_dot_product_attention")
    rng = np.random.default_rng(1)
    T, d = 5, 8
    x = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(rng, d)
    got = mha(x, Wq, Wk, Wv, Wo, heads=1)
    want = sdpa(x @ Wq, x @ Wk, x @ Wv)[0] @ Wo
    close(got, want, atol=1e-9, label="one-head output")


@check("the worked example gives rows [0.6698, 0.3302, 1, 1] and [0.3302, 0.6698, 1, 1]")
def test_worked_example(sub):
    mha = has(sub, "multi_head_attention")
    x = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    eye = np.eye(4)
    got = mha(x, eye, eye, eye, eye, heads=2)
    want = np.array([[0.66976155, 0.33023845, 1.0, 1.0], [0.33023845, 0.66976155, 1.0, 1.0]])
    close(got, want, atol=1e-6, label="output of the two-head example")


@check("two and four heads match a manual head-by-head computation, masked and unmasked")
def test_matches_manual(sub):
    mha = has(sub, "multi_head_attention")
    causal = has(sub, "causal_mask")
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


@check("with the causal mask, changing later tokens never changes earlier outputs, in every head")
def test_causal_independence(sub):
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


@check("without a mask, permuting the input rows permutes the output rows the same way")
def test_permutation(sub):
    mha = has(sub, "multi_head_attention")
    rng = np.random.default_rng(4)
    T, d = 6, 8
    x = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(rng, d)
    perm = rng.permutation(T)
    out = mha(x, Wq, Wk, Wv, Wo, heads=2)
    out_perm = mha(x[perm], Wq, Wk, Wv, Wo, heads=2)
    close(out_perm, out[perm], atol=1e-9, label="output of the shuffled input")
