"""Checks for the scaling-the-scores exercise. The softmax is provided, so only the new functions are tested."""

import numpy as np

from mlcheck import check, close, fail, has, shape


@check("attention_score_scale is 1 / sqrt(d_k)")
def test_scale(sub):
    fn = has(sub, "attention_score_scale")
    for d_k, want in [(1, 1.0), (4, 0.5), (16, 0.25), (64, 0.125), (256, 0.0625)]:
        got = fn(d_k)
        if abs(float(got) - want) > 1e-12:
            fail(f"attention_score_scale({d_k}) is {got}, expected {want}")


@check("score_std is close to sqrt(d_k) for random unit-variance vectors")
def test_score_std(sub):
    fn = has(sub, "score_std")
    for d_k in (4, 16, 64, 256):
        got = fn(d_k)
        want = np.sqrt(d_k)
        if abs(got - want) > 0.10 * want:
            fail(f"score_std({d_k}) is {got:.3f}, expected about {want:.3f} (within 10 percent)")
    if fn(16, seed=5) != fn(16, seed=5):
        fail("score_std must be repeatable for the same seed")


@check("dividing the raw spread by sqrt(d_k) leaves a spread near 1 at every width")
def test_scaled_spread(sub):
    std = has(sub, "score_std")
    scale = has(sub, "attention_score_scale")
    for d_k in (4, 16, 64, 256):
        got = std(d_k, n=2000, seed=1) * scale(d_k)
        if abs(got - 1.0) > 0.10:
            fail(f"spread after scaling at d_k={d_k} is {got:.3f}, expected about 1")
    ratio = std(256) / std(4)
    if abs(ratio - 8.0) > 0.9:
        fail(f"std at d_k=256 over std at d_k=4 is {ratio:.2f}; the square-root law predicts 8")


@check("scaled_dot_product_attention returns the right shapes and rows that sum to 1")
def test_shapes(sub):
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


@check("the worked 64-wide example gives weights [0.7311, 0.2689] and [0.6225, 0.3775]")
def test_hand_example(sub):
    fn = has(sub, "scaled_dot_product_attention")
    q = np.array([[0.25] * 64, [0.125] * 64])
    k = np.array([[1.0] * 64, [0.5] * 64])
    out, weights = fn(q, k, np.eye(2))
    want = np.array([[0.7310585786, 0.2689414214], [0.6224593312, 0.3775406688]])
    close(weights, want, atol=1e-8, label="weights of the 64-wide example")
    close(out, want, atol=1e-8, label="output with identity values")


@check("the output equals the explicit softmax(q k^T / sqrt(d_k)) v")
def test_matches_formula(sub):
    fn = has(sub, "scaled_dot_product_attention")
    rng = np.random.default_rng(7)
    q = rng.normal(size=(5, 9))
    k = rng.normal(size=(5, 9))
    v = rng.normal(size=(5, 4))
    s = q @ k.T / 3.0
    s = s - s.max(axis=1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(axis=1, keepdims=True)
    out, weights = fn(q, k, v)
    close(weights, w, atol=1e-10, label="weights")
    close(out, w @ v, atol=1e-10, label="output")


@check("attention stays unsaturated at width 256: no row is close to one-hot")
def test_not_saturated(sub):
    fn = has(sub, "scaled_dot_product_attention")
    rng = np.random.default_rng(3)
    q = rng.normal(size=(40, 256))
    k = rng.normal(size=(8, 256))
    _, weights = fn(q, k, np.eye(8))
    mean_max = float(weights.max(axis=1).mean())
    if mean_max > 0.5:
        fail(f"the average largest weight is {mean_max:.3f}; scaled scores at d_k=256 should give about 0.36")
