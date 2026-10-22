"""Checks for the attention scores and weights exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


@check("dot_scores gives the table of query-key dot products, with no scaling")
def test_dot_scores_hand(sub):
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


@check("dot_scores has shape (T_q, T_k) and entry [i, j] is query i dotted with key j")
def test_dot_scores_shape(sub):
    fn = has(sub, "dot_scores")
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


@check("softmax of [2, 1] is [0.7311, 0.2689]")
def test_softmax_hand(sub):
    fn = has(sub, "softmax")
    close(fn(np.array([2.0, 1.0])), np.array([0.7310585786, 0.2689414214]), atol=1e-8,
          label="softmax([2, 1])")


@check("softmax stays finite at a score of 900 and ignores a constant shift")
def test_softmax_stable(sub):
    fn = has(sub, "softmax")
    out = fn(np.array([900.0, 0.0]))
    if not np.all(np.isfinite(out)):
        fail(f"softmax([900, 0]) returned non-finite values {out}")
    close(out, np.array([1.0, 0.0]), atol=1e-12, label="softmax([900, 0])")
    z = np.array([0.3, -1.2, 2.0, 0.0])
    close(fn(z + 1000.0), fn(z), atol=1e-10, label="softmax after adding 1000 to every score")


@check("softmax shifts each row by its own maximum, so rows of very different size all stay finite")
def test_softmax_rowwise_shift(sub):
    fn = has(sub, "softmax")
    x = np.array([[900.0, 0.0], [0.0, -900.0], [-1000.0, -1001.0]])
    out = fn(x, axis=-1)
    if not np.all(np.isfinite(out)):
        fail(f"softmax returned non-finite values for rows of different size: {out}; "
             "subtract the maximum of each row, not one maximum for the whole array")
    close(out[2], np.array([0.7310585786, 0.2689414214]), atol=1e-8, label="row [-1000, -1001]")
    close(out[:2], np.array([[1.0, 0.0], [1.0, 0.0]]), atol=1e-12, label="first two rows")


@check("softmax rows sum to 1 and the axis argument picks rows or columns")
def test_softmax_axis(sub):
    fn = has(sub, "softmax")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 4)) * 5
    rows = fn(x, axis=-1)
    shape(rows, (3, 4), label="softmax output")
    close(rows.sum(axis=1), np.ones(3), atol=1e-10, label="row sums")
    cols = fn(x, axis=0)
    close(cols.sum(axis=0), np.ones(4), atol=1e-10, label="column sums")
    if np.any(rows <= 0):
        fail("every softmax weight must be strictly positive for moderate scores")


@check("blend_values with one-hot weights returns a row of v exactly")
def test_blend_onehot(sub):
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


@check("blend_values of uniform weights is the average of the value rows")
def test_blend_average(sub):
    fn = has(sub, "blend_values")
    rng = np.random.default_rng(3)
    v = rng.normal(size=(5, 2))
    weights = np.full((1, 5), 0.2)
    close(fn(weights, v), v.mean(axis=0, keepdims=True), atol=1e-12, label="uniform blend")


@check("score, softmax, and blend together reproduce the worked example [0.7311, 0.2689]")
def test_pipeline(sub):
    q = np.array([[1.0, 1.0]])
    k = np.array([[1.5, 0.5], [0.5, 0.5]])
    v = np.eye(2)
    scores = has(sub, "dot_scores")(q, k)
    weights = has(sub, "softmax")(scores, axis=-1)
    out = has(sub, "blend_values")(weights, v)
    close(out, np.array([[0.7310585786, 0.2689414214]]), atol=1e-8, label="blended output")
