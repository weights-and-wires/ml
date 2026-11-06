"""Checks for the alignment-matrix and copy-decoder exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def random_case(seed, T_y=4, T_x=5, d_s=3, d_h=4, h=6, d_v=2):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(T_y, d_s)),
        rng.normal(size=(T_x, d_h)),
        rng.normal(size=(T_x, d_v)),
        0.6 * rng.normal(size=(h, d_s)),
        0.6 * rng.normal(size=(h, d_h)),
        0.6 * rng.normal(size=h),
    )


@check("alignment_matrix has one row per decoder step, and every row sums to one")
def test_alignment_shape(sub):
    fn = has(sub, "alignment_matrix")
    queries, keys, values, W1, W2, v = random_case(0)
    contexts, A = fn(queries, keys, values, W1, W2, v)
    shape(contexts, (4, 2), label="contexts")
    shape(A, (4, 5), label="alignment matrix")
    if np.any(A < 0):
        fail("every attention weight must be non-negative")
    close(A.sum(axis=1), np.ones(4), atol=1e-9, label="row sums of the alignment matrix")


@check("alignment_matrix equals stacking one additive_attention call per query")
def test_alignment_stacks(sub):
    fn = has(sub, "alignment_matrix")
    attention = has(sub, "additive_attention")
    queries, keys, values, W1, W2, v = random_case(1)
    contexts, A = fn(queries, keys, values, W1, W2, v)
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


@check("decode_with_attention copies the source [3, 0, 4, 1, 2] exactly")
def test_decode_copies(sub):
    fn = has(sub, "decode_with_attention")
    source = [3, 0, 4, 1, 2]
    decoded, alignment = fn(source, vocab_size=5)
    if list(decoded) != source:
        fail(f"the copy decoder should reproduce the source: got {list(decoded)}, wanted {source}")
    shape(alignment, (5, 5), label="alignment matrix")


@check("the alignment matrix of the copy decoder is within 0.01 of the identity")
def test_decode_alignment(sub):
    fn = has(sub, "decode_with_attention")
    _, alignment = fn([4, 2, 0, 3, 1], vocab_size=5)
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
