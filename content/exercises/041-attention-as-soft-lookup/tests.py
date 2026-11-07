"""Checks for the soft-lookup exercise: dot-product and additive attention."""

import numpy as np

from mlcheck import check, close, fail, has, shape

QUERY = np.array([1.0, 0.0])
KEYS = np.array([[1.0, 0.0], [0.0, 1.0]])
VALUES = np.array([[10.0, 0.0], [0.0, 10.0]])
EYE = np.eye(2)


def ref_softmax(z):
    e = np.exp(np.asarray(z, dtype=float) - np.max(z))
    return e / e.sum()


def random_case(seed, T=6, d_s=3, d_h=4, h=5, d_v=2):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=d_s),
        rng.normal(size=(T, d_h)),
        rng.normal(size=(T, d_v)),
        0.7 * rng.normal(size=(h, d_s)),
        0.7 * rng.normal(size=(h, d_h)),
        0.7 * rng.normal(size=h),
    )


@check("softmax turns [1, 0] into 0.7311 and 0.2689, survives scores of 800, and ignores a shift")
def test_softmax(sub):
    softmax = has(sub, "softmax")
    out = softmax(np.array([1.0, 0.0]))
    shape(out, (2,), label="softmax output")
    close(out, np.array([0.7310586, 0.2689414]), atol=1e-6, label="softmax([1, 0])")
    out = softmax(np.array([-800.0, 0.0, 800.0]))
    if not np.all(np.isfinite(out)):
        fail(f"softmax returned non-finite values for scores [-800, 0, 800]: {out}")
    close(out, np.array([0.0, 0.0, 1.0]), atol=1e-12, label="softmax([-800, 0, 800])")
    close(
        softmax(np.array([101.0, 100.0])),
        softmax(np.array([1.0, 0.0])),
        atol=1e-12,
        label="softmax([101, 100]) against softmax([1, 0])",
    )


@check("attend reproduces the worked example: weights [0.7311, 0.2689], context [7.311, 2.689]")
def test_attend_worked(sub):
    attend = has(sub, "attend")
    context, weights = attend(np.array([1.0, 0.0]), VALUES)
    shape(context, (2,), label="context")
    shape(weights, (2,), label="weights")
    close(weights, np.array([0.7310586, 0.2689414]), atol=1e-6, label="weights")
    close(context, np.array([7.310586, 2.689414]), atol=1e-5, label="context")


@check("attend returns weights @ values, inside the range of the values, and nearly a hard pick for a big gap")
def test_attend_convex(sub):
    attend = has(sub, "attend")
    rng = np.random.default_rng(0)
    scores = rng.normal(size=6) * 3
    values = rng.normal(size=(6, 4))
    context, weights = attend(scores, values)
    shape(context, (4,), label="context")
    close(weights.sum(), 1.0, atol=1e-10, label="sum of the weights")
    close(context, weights @ values, atol=1e-10, label="context as weights @ values")
    lo, hi = values.min(axis=0), values.max(axis=0)
    if np.any(context < lo - 1e-9) or np.any(context > hi + 1e-9):
        fail(f"a blend cannot leave the range of its values [{lo}, {hi}], but the context is {context}")
    picks = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 4.0]])
    sharp, _ = attend(np.array([40.0, 0.0, 0.0]), picks)
    close(sharp, picks[0], atol=1e-6, label="context for scores [40, 0, 0]")


@check("multiplicative_attention with no matrix scores by dot product, and W = identity changes nothing")
def test_dot_form(sub):
    fn = has(sub, "multiplicative_attention")
    context, _ = fn(QUERY, KEYS, VALUES)
    close(context, np.array([7.310586, 2.689414]), atol=1e-5, label="running-example context")

    rng = np.random.default_rng(1)
    query = rng.normal(size=4)
    keys = rng.normal(size=(5, 4))
    values = rng.normal(size=(5, 3))
    c0, w0 = fn(query, keys, values)
    close(w0, ref_softmax([query @ k for k in keys]), atol=1e-8, label="weights from dot products")
    c1, w1 = fn(query, keys, values, np.eye(4))
    close(w1, w0, atol=1e-10, label="weights with W = identity")
    close(c1, c0, atol=1e-10, label="context with W = identity")


@check("with a non-square W, score i is query @ W @ keys[i]")
def test_general_form(sub):
    fn = has(sub, "multiplicative_attention")
    rng = np.random.default_rng(3)
    query = rng.normal(size=3)
    keys = rng.normal(size=(6, 5))
    values = rng.normal(size=(6, 2))
    W = rng.normal(size=(3, 5))
    context, weights = fn(query, keys, values, W)
    shape(weights, (6,), label="weights")
    shape(context, (2,), label="context")
    want = ref_softmax([query @ W @ k for k in keys])
    close(weights, want, atol=1e-8, label="weights from query @ W @ keys[i]")


@check("additive_scores gives [0.9640, 1.5232] on the worked example and matches a loop over keys")
def test_scores(sub):
    fn = has(sub, "additive_scores")
    scores = fn(QUERY, KEYS, EYE, EYE, np.array([1.0, 1.0]))
    shape(scores, (2,), label="scores")
    close(scores, np.array([0.9640276, 1.5231883]), atol=1e-5, label="scores")
    query, keys, _, W1, W2, v = random_case(0)
    scores = fn(query, keys, W1, W2, v)
    shape(scores, (6,), label="scores")
    want = np.array([v @ np.tanh(W1 @ query + W2 @ k) for k in keys])
    close(scores, want, atol=1e-10, label="scores against v @ tanh(W1 @ query + W2 @ keys[i])")


@check("additive_attention gives weights [0.3637, 0.6363], flipping v swaps them, and the opposite key can win")
def test_additive_attention(sub):
    fn = has(sub, "additive_attention")
    context, weights = fn(QUERY, KEYS, VALUES, EYE, EYE, np.array([1.0, 1.0]))
    close(weights, np.array([0.3637, 0.6363]), atol=1e-4, label="weights")
    close(context, np.array([3.6374, 6.3626]), atol=1e-3, label="context")
    _, flipped = fn(QUERY, KEYS, VALUES, EYE, EYE, np.array([-1.0, -1.0]))
    close(flipped, np.array([0.6363, 0.3637]), atol=1e-4, label="weights with v = [-1, -1]")

    keys = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _, weights = fn(QUERY, keys, VALUES, EYE, EYE, np.array([-1.0, -1.0]))
    # Scores are -0.964 for the identical key and 0.0 for the opposite one.
    close(weights, ref_softmax([-0.9640276, 0.0]), atol=1e-4, label="weights for keys [1, 0] and [-1, 0]")

    query, keys, values, W1, W2, v = random_case(1)
    context, weights = fn(query, keys, values, W1, W2, v)
    shape(weights, (6,), label="weights")
    close(context, weights @ values, atol=1e-10, label="context as weights @ values")
    close(weights, ref_softmax(np.tanh(W1 @ query + keys @ W2.T) @ v), atol=1e-10,
          label="weights as softmax of the additive scores")


@check("emulate_dot_params returns c * identity twice and query / c")
def test_emulate_shapes(sub):
    fn = has(sub, "emulate_dot_params")
    query = np.array([0.5, -1.0, 2.0])
    W1, W2, v = fn(query, 0.01)
    shape(W1, (3, 3), label="W1")
    shape(W2, (3, 3), label="W2")
    shape(v, (3,), label="v")
    close(W1, 0.01 * np.eye(3), atol=1e-12, label="W1 = c * identity")
    close(W2, 0.01 * np.eye(3), atol=1e-12, label="W2 = c * identity")
    close(v, query / 0.01, atol=1e-9, label="v = query / c")


@check("with those parameters, additive attention gives the same weights as a plain dot product")
def test_emulates_dot(sub):
    emulate = has(sub, "emulate_dot_params")
    additive = has(sub, "additive_attention")
    rng = np.random.default_rng(5)
    query = rng.normal(size=4)
    keys = rng.normal(size=(7, 4))
    values = rng.normal(size=(7, 3))
    W1, W2, v = emulate(query, 1e-3)
    _, w_add = additive(query, keys, values, W1, W2, v)
    close(w_add, ref_softmax(keys @ query), atol=1e-4, label="additive weights against softmax(keys @ query)")


@check("parameters built for one query keep following that query when the query changes")
def test_emulation_is_query_specific(sub):
    emulate = has(sub, "emulate_dot_params")
    additive = has(sub, "additive_attention")
    rng = np.random.default_rng(0)
    query = rng.normal(size=3)
    other = rng.normal(size=3)
    keys = rng.normal(size=(5, 3))
    values = rng.normal(size=(5, 2))
    W1, W2, v = emulate(query, 1e-3)
    _, w_other = additive(other, keys, values, W1, W2, v)
    _, w_built = additive(query, keys, values, W1, W2, v)
    close(w_other, w_built, atol=1e-4, label="weights for a different query against weights for the built query")
    gap = np.abs(w_other - ref_softmax(keys @ other)).max()
    if gap < 0.05:
        fail(f"with no bend the weights should ignore the new query, but they sit within {gap:.3f} of its dot-product weights")
