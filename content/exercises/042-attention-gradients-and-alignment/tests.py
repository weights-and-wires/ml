"""Checks for the attention-gradients and alignment-matrix exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape

QUERY = np.array([1.0, 0.0])
KEYS = np.array([[1.0, 0.0], [0.0, 1.0]])
VALUES = np.array([[10.0, 0.0], [0.0, 10.0]])


def ref_softmax(z):
    e = np.exp(np.asarray(z, dtype=float) - np.max(z))
    return e / e.sum()


def random_case(seed, T=5, d_s=3, d_h=4, h=6, d_v=2):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=d_s),
        rng.normal(size=(T, d_h)),
        rng.normal(size=(T, d_v)),
        0.6 * rng.normal(size=(h, d_s)),
        0.6 * rng.normal(size=(h, d_h)),
        0.6 * rng.normal(size=h),
    )


def random_alignment_case(seed, T_y=4, T_x=5, d_s=3, d_h=4, h=6, d_v=2):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(T_y, d_s)),
        rng.normal(size=(T_x, d_h)),
        rng.normal(size=(T_x, d_v)),
        0.6 * rng.normal(size=(h, d_s)),
        0.6 * rng.normal(size=(h, d_h)),
        0.6 * rng.normal(size=h),
    )


@check("softmax_backward gives [2.314, -2.314] on the worked example, sums to zero, and is zero for a constant slope")
def test_softmax_backward_worked(sub):
    fn = has(sub, "softmax_backward")
    weights = ref_softmax([0.9640276, 1.5231883])
    dscores = fn(weights, np.array([10.0, 0.0]))
    shape(dscores, (2,), label="dscores")
    close(dscores, np.array([2.3143, -2.3143]), atol=1e-3, label="dscores")
    close(dscores.sum(), 0.0, atol=1e-10, label="sum of dscores")
    weights = ref_softmax([0.3, -1.0, 2.0, 0.5])
    close(fn(weights, np.full(4, 7.0)), np.zeros(4), atol=1e-12, label="dscores for a constant dweights")


@check("softmax_backward matches finite differences of softmax(z) @ dweights")
def test_softmax_backward_fd(sub):
    fn = has(sub, "softmax_backward")
    rng = np.random.default_rng(0)
    z = rng.normal(size=6)
    dweights = rng.normal(size=6)
    got = fn(ref_softmax(z), dweights)
    want = finite_difference(lambda arr: float(ref_softmax(arr) @ dweights), z)
    close(got, want, atol=1e-6, rtol=1e-5, label="dscores against finite differences")


@check("context_backward gives dweights [10, 0] on the worked example and matches finite differences")
def test_context_backward(sub):
    fn = has(sub, "context_backward")
    weights = ref_softmax([0.9640276, 1.5231883])
    dweights, dvalues = fn(np.array([1.0, 0.0]), weights, VALUES)
    shape(dweights, (2,), label="dweights")
    shape(dvalues, (2, 2), label="dvalues")
    close(dweights, np.array([10.0, 0.0]), atol=1e-9, label="dweights")
    close(dvalues, np.array([[0.3637, 0.0], [0.6363, 0.0]]), atol=1e-4, label="dvalues")

    rng = np.random.default_rng(1)
    weights = ref_softmax(rng.normal(size=5))
    values = rng.normal(size=(5, 3))
    dcontext = rng.normal(size=3)
    dweights, dvalues = fn(dcontext, weights, values)
    fd_w = finite_difference(lambda w: float((w @ values) @ dcontext), weights)
    fd_v = finite_difference(lambda x: float((weights @ x) @ dcontext), values)
    close(dweights, fd_w, atol=1e-6, rtol=1e-5, label="dweights against finite differences")
    close(dvalues, fd_v, atol=1e-6, rtol=1e-5, label="dvalues against finite differences")


@check("attention_backward matches finite differences on all six inputs")
def test_attention_backward_fd(sub):
    forward = has(sub, "additive_attention")
    backward = has(sub, "attention_backward")
    query, keys, values, W1, W2, v = random_case(2)
    dcontext = np.random.default_rng(20).normal(size=values.shape[1])
    grads = backward(dcontext, query, keys, values, W1, W2, v)
    names = ["query", "keys", "values", "W1", "W2", "v"]
    args = {"query": query, "keys": keys, "values": values, "W1": W1, "W2": W2, "v": v}
    if len(grads) != 6:
        fail(f"attention_backward must return 6 gradients, got {len(grads)}")
    for name, grad in zip(names, grads):
        def loss(arr, name=name):
            local = dict(args)
            local[name] = arr
            context, _ = forward(local["query"], local["keys"], local["values"],
                                 local["W1"], local["W2"], local["v"])
            return float(context @ dcontext)

        shape(grad, args[name].shape, label=f"gradient for {name}")
        close(grad, finite_difference(loss, args[name]), atol=1e-6, rtol=1e-4,
              label=f"gradient for {name}")


@check("with one key the weight is always 1, so only the values receive a gradient")
def test_single_key(sub):
    backward = has(sub, "attention_backward")
    query, keys, values, W1, W2, v = random_case(3, T=1)
    dcontext = np.array([0.5, -2.0])
    dquery, dkeys, dvalues, dW1, dW2, dv = backward(dcontext, query, keys, values, W1, W2, v)
    close(dvalues, dcontext[None, :], atol=1e-12, label="dvalues")
    for name, grad in (("dquery", dquery), ("dkeys", dkeys), ("dW1", dW1), ("dW2", dW2), ("dv", dv)):
        close(grad, np.zeros_like(grad), atol=1e-12, label=name)


@check("alignment_matrix has one row per step, each row sums to one, and each row is one additive_attention call")
def test_alignment_matrix(sub):
    fn = has(sub, "alignment_matrix")
    attention = has(sub, "additive_attention")
    queries, keys, values, W1, W2, v = random_alignment_case(0)
    contexts, A = fn(queries, keys, values, W1, W2, v)
    shape(contexts, (4, 2), label="contexts")
    shape(A, (4, 5), label="alignment matrix")
    if np.any(A < 0):
        fail("every attention weight must be non-negative")
    close(A.sum(axis=1), np.ones(4), atol=1e-9, label="row sums of the alignment matrix")
    for t, query in enumerate(queries):
        c, w = attention(query, keys, values, W1, W2, v)
        close(A[t], w, atol=1e-10, label=f"row {t} of the alignment matrix")
        close(contexts[t], c, atol=1e-10, label=f"context {t}")


@check("mean_summary is the same for every ordering of the same tokens")
def test_mean_summary(sub):
    fn = has(sub, "mean_summary")
    one = fn([0, 1, 2], 3)
    two = fn([2, 1, 0], 3)
    shape(one, (3,), label="summary")
    close(one, np.full(3, 1 / 3), atol=1e-12, label="summary of [0, 1, 2]")
    close(two, one, atol=1e-12, label="summary of [2, 1, 0] against [0, 1, 2]")
    other = fn([0, 0, 2], 4)
    shape(other, (4,), label="summary with vocab_size 4")
    close(other, np.array([2 / 3, 0.0, 1 / 3, 0.0]), atol=1e-12, label="summary of [0, 0, 2]")


@check("decode_with_attention copies [3, 0, 4, 1, 2] and its alignment matrix is within 0.01 of the identity")
def test_decode_copies(sub):
    fn = has(sub, "decode_with_attention")
    source = [3, 0, 4, 1, 2]
    decoded, alignment = fn(source, vocab_size=5)
    if list(decoded) != source:
        fail(f"the copy decoder should reproduce the source: got {list(decoded)}, wanted {source}")
    shape(alignment, (5, 5), label="alignment matrix")
    close(alignment, np.eye(5), atol=1e-2, rtol=0.0, label="alignment against the identity")


@check("with the sign of v flipped, the decoder looks everywhere except the matching position")
def test_flipped_sign(sub):
    fn = has(sub, "decode_with_attention")
    source = [3, 0, 4, 1, 2]
    decoded, alignment = fn(source, vocab_size=5, v_sign=1.0)
    diagonal = np.diag(alignment)
    if np.any(diagonal > 0.01):
        fail(f"with v_sign = +1 the diagonal weights should be below 0.01, got {diagonal.round(4)}")
    off = alignment[~np.eye(5, dtype=bool)].reshape(5, 4)
    close(off, np.full((5, 4), 0.25), atol=1e-2, rtol=0.0, label="off-diagonal weights, about 1/(T-1) each")
    if list(decoded) == source:
        fail("a flipped v should no longer reproduce the source")


@check("attention tells [0, 1, 2] from [2, 1, 0], while the mean summary cannot")
def test_order(sub):
    decode = has(sub, "decode_with_attention")
    summary = has(sub, "mean_summary")
    first, _ = decode([0, 1, 2], vocab_size=3)
    second, _ = decode([2, 1, 0], vocab_size=3)
    if list(first) != [0, 1, 2] or list(second) != [2, 1, 0]:
        fail(f"attention should keep the order: got {list(first)} and {list(second)}")
    close(summary([0, 1, 2], 3), summary([2, 1, 0], 3), atol=1e-12, label="the two mean summaries")
