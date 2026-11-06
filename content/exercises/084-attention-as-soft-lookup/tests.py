"""Checks for the soft-lookup exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape

QUERY = np.array([1.0, 0.0])
KEYS = np.array([[1.0, 0.0], [0.0, 1.0]])
VALUES = np.array([[10.0, 0.0], [0.0, 10.0]])


def ref_softmax(z):
    e = np.exp(np.asarray(z, dtype=float) - np.max(z))
    return e / e.sum()


@check("softmax turns the scores [1, 0] into the weights 0.7311 and 0.2689")
def test_softmax_worked(sub):
    softmax = has(sub, "softmax")
    out = softmax(np.array([1.0, 0.0]))
    shape(out, (2,), label="softmax output")
    close(out, np.array([0.7310586, 0.2689414]), atol=1e-6, label="softmax([1, 0])")


@check("softmax survives huge scores and ignores a shift in all scores")
def test_softmax_stable(sub):
    softmax = has(sub, "softmax")
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


@check("attend reproduces the worked example: context [7.311, 2.689]")
def test_attend_worked(sub):
    attend = has(sub, "attend")
    context, weights = attend(np.array([1.0, 0.0]), VALUES)
    shape(context, (2,), label="context")
    shape(weights, (2,), label="weights")
    close(weights, np.array([0.7310586, 0.2689414]), atol=1e-6, label="weights")
    close(context, np.array([7.310586, 2.689414]), atol=1e-5, label="context")


@check("attend returns weights @ values, which stays inside the range of the values")
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


@check("attend turns almost into a hard pick when one score is far above the rest")
def test_attend_sharp(sub):
    attend = has(sub, "attend")
    values = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 4.0]])
    context, _ = attend(np.array([40.0, 0.0, 0.0]), values)
    close(context, values[0], atol=1e-6, label="context for scores [40, 0, 0]")


@check("multiplicative_attention with no matrix scores each key by its dot product")
def test_dot_form(sub):
    fn = has(sub, "multiplicative_attention")
    context, weights = fn(QUERY, KEYS, VALUES)
    close(context, np.array([7.310586, 2.689414]), atol=1e-5, label="running-example context")

    rng = np.random.default_rng(1)
    query = rng.normal(size=4)
    keys = rng.normal(size=(5, 4))
    values = rng.normal(size=(5, 3))
    _, weights = fn(query, keys, values)
    want = ref_softmax([query @ k for k in keys])
    close(weights, want, atol=1e-8, label="weights from dot products")


@check("W equal to the identity gives the same answer as no W")
def test_identity_matrix(sub):
    fn = has(sub, "multiplicative_attention")
    rng = np.random.default_rng(2)
    query = rng.normal(size=4)
    keys = rng.normal(size=(5, 4))
    values = rng.normal(size=(5, 3))
    c0, w0 = fn(query, keys, values)
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
