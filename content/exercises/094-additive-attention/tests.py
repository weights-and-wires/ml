"""Checks for the additive-attention exercise."""

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


@check("additive_scores gives [0.9640, 1.5232] on the worked example")
def test_scores_worked(sub):
    fn = has(sub, "additive_scores")
    scores = fn(QUERY, KEYS, EYE, EYE, np.array([1.0, 1.0]))
    shape(scores, (2,), label="scores")
    close(scores, np.array([0.9640276, 1.5231883]),
          atol=1e-5, label="scores")


@check("additive_scores matches a loop over keys, with different widths for query and keys")
def test_scores_loop(sub):
    fn = has(sub, "additive_scores")
    query, keys, _, W1, W2, v = random_case(0)
    scores = fn(query, keys, W1, W2, v)
    shape(scores, (6,), label="scores")
    want = np.array([v @ np.tanh(W1 @ query + W2 @ k) for k in keys])
    close(scores, want, atol=1e-10, label="scores against v @ tanh(W1 @ query + W2 @ keys[i])")


@check("additive_attention gives weights [0.3637, 0.6363] and context [3.637, 6.363], and flipping v swaps the weights")
def test_attention_worked(sub):
    fn = has(sub, "additive_attention")
    context, weights = fn(QUERY, KEYS, VALUES, EYE, EYE, np.array([1.0, 1.0]))
    close(weights, np.array([0.3637, 0.6363]), atol=1e-4, label="weights")
    close(context, np.array([3.6374, 6.3626]), atol=1e-3, label="context")
    _, flipped = fn(QUERY, KEYS, VALUES, EYE, EYE, np.array([-1.0, -1.0]))
    close(flipped, np.array([0.6363, 0.3637]), atol=1e-4, label="weights with v = [-1, -1]")


@check("the weights form a distribution and the context is their blend of the values")
def test_distribution(sub):
    fn = has(sub, "additive_attention")
    query, keys, values, W1, W2, v = random_case(1)
    context, weights = fn(query, keys, values, W1, W2, v)
    shape(weights, (6,), label="weights")
    shape(context, (2,), label="context")
    if np.any(weights < 0) or abs(weights.sum() - 1.0) > 1e-9:
        fail(f"weights must be non-negative and sum to 1, got sum {weights.sum():.6f}")
    close(context, weights @ values, atol=1e-10, label="context as weights @ values")
    close(weights, ref_softmax(np.tanh(W1 @ query + keys @ W2.T) @ v), atol=1e-10,
          label="weights as softmax of the additive scores")


@check("a key that is the opposite of the query can win: v decides what counts as a match")
def test_opposite_key(sub):
    fn = has(sub, "additive_attention")
    keys = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _, weights = fn(QUERY, keys, VALUES, EYE, EYE, np.array([-1.0, -1.0]))
    # Scores are -0.964 for the identical key and 0.0 for the opposite one.
    close(weights, ref_softmax([-0.9640276, 0.0]), atol=1e-4, label="weights for keys [1, 0] and [-1, 0]")


@check("emulate_dot_params returns the right shapes and values")
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
    want = ref_softmax(keys @ query)
    close(w_add, want, atol=1e-4, label="additive weights against softmax(keys @ query)")


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
