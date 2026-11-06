"""Checks for the rotary embeddings exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def explicit_rotation(x, p, d, theta=10000.0):
    """Turn each pair (j, j + d/2) of a single vector x by p * theta ** (-2j / d)."""
    out = np.array(x, dtype=float)
    half = d // 2
    for j in range(half):
        a = p * theta ** (-2.0 * j / d)
        u, v = x[j], x[j + half]
        out[j] = u * np.cos(a) - v * np.sin(a)
        out[j + half] = u * np.sin(a) + v * np.cos(a)
    return out


@check("rope_tables returns (T, d) tables, with row 0 equal to (1, 0) and cos^2 + sin^2 = 1")
def test_tables_basic(sub):
    cos, sin = has(sub, "rope_tables")(T=6, d=8)
    shape(cos, (6, 8), label="cos table")
    shape(sin, (6, 8), label="sin table")
    close(cos[0], np.ones(8), atol=1e-12, label="cos at position 0")
    close(sin[0], np.zeros(8), atol=1e-12, label="sin at position 0")
    close(cos**2 + sin**2, np.ones((6, 8)), atol=1e-12, label="cos^2 + sin^2")


@check("rope_tables stores each speed theta ** (-2j / d) in both halves")
def test_tables_values(sub):
    d, T, theta = 8, 5, 10000.0
    cos, sin = has(sub, "rope_tables")(T, d, theta)
    for j in range(d // 2):
        speed = theta ** (-2.0 * j / d)
        want_cos = np.cos(np.arange(T) * speed)
        want_sin = np.sin(np.arange(T) * speed)
        close(cos[:, j], want_cos, atol=1e-10, label=f"cos column {j}")
        close(cos[:, j + d // 2], want_cos, atol=1e-10, label=f"cos column {j + d // 2}")
        close(sin[:, j], want_sin, atol=1e-10, label=f"sin column {j}")
        close(sin[:, j + d // 2], want_sin, atol=1e-10, label=f"sin column {j + d // 2}")
    cos2, _ = has(sub, "rope_tables")(3, 4, 100.0)
    close(cos2[:, 1], np.cos(np.arange(3) * 100.0 ** (-0.5)), atol=1e-10, label="a different theta")


@check("apply_rope turns [1, 0] at position 1 to [cos 1, sin 1] and matches an explicit turn of each pair")
def test_apply_hand(sub):
    tables = has(sub, "rope_tables")
    apply_rope = has(sub, "apply_rope")
    cos, sin = tables(3, 2)
    out = apply_rope(np.array([1.0, 0.0]), cos[1], sin[1])
    close(out, np.array([np.cos(1.0), np.sin(1.0)]), atol=1e-10, label="[1, 0] at position 1, width 2")
    rng = np.random.default_rng(0)
    x = rng.normal(size=8)
    cos, sin = tables(10, 8)
    for p in (0, 3, 9):
        close(apply_rope(x, cos[p], sin[p]), explicit_rotation(x, p, 8), atol=1e-9, label=f"width 8 at position {p}")


@check("apply_rope keeps lengths and works on a vector, a sequence, and a stack of heads")
def test_apply_shapes(sub):
    tables = has(sub, "rope_tables")
    apply_rope = has(sub, "apply_rope")
    rng = np.random.default_rng(1)
    T, d, heads = 7, 12, 3
    cos, sin = tables(T, d)
    seq = rng.normal(size=(T, d))
    out = apply_rope(seq, cos, sin)
    shape(out, (T, d), label="sequence output")
    close(np.linalg.norm(out, axis=-1), np.linalg.norm(seq, axis=-1), atol=1e-9, label="row lengths")
    stack = rng.normal(size=(heads, T, d))
    out3 = apply_rope(stack, cos, sin)
    shape(out3, (heads, T, d), label="stack output")
    for h in range(heads):
        close(out3[h], apply_rope(stack[h], cos, sin), atol=1e-12, label=f"head {h}")
    for p in (0, 4):
        close(out[p], explicit_rotation(seq[p], p, d), atol=1e-9, label=f"row {p} against an explicit turn")


@check("rope_score equals cos(n - m) for q = k = [1, 0] at width 2, and the plain dot product at m = n")
def test_score_hand(sub):
    score = has(sub, "rope_score")
    q = k = np.array([1.0, 0.0])
    close(score(q, k, 0, 1), np.cos(1.0), atol=1e-10, label="positions 0 and 1")
    close(score(q, k, 1, 2), np.cos(1.0), atol=1e-10, label="positions 1 and 2")
    close(score(q, k, 1, 3), np.cos(2.0), atol=1e-10, label="positions 1 and 3")
    rng = np.random.default_rng(2)
    q, k = rng.normal(size=10), rng.normal(size=10)
    for m in (0, 6, 30):
        close(score(q, k, m, m), float(q @ k), atol=1e-9, label=f"plain dot product at m = n = {m}")


@check("rope_score depends only on the offset n - m, and changes when the offset changes")
def test_score_offset(sub):
    score = has(sub, "rope_score")
    rng = np.random.default_rng(3)
    d = 16
    q, k = rng.normal(size=d), rng.normal(size=d)
    for offset in (-4, 0, 3, 11):
        values = [score(q, k, m, m + offset) for m in (5, 9, 20, 47) if m + offset >= 0]
        spread = max(values) - min(values)
        if spread > 1e-8:
            fail(f"at offset {offset} the score should not depend on the position, but it varied by {spread:.3g}: {values}")
    if abs(score(q, k, 0, 3) - score(q, k, 0, 4)) < 1e-6:
        fail("changing the offset from 3 to 4 should change the score, but it did not")
    close(score(q, k, 2, 9, 500.0), float(explicit_rotation(q, 2, d, 500.0) @ explicit_rotation(k, 9, d, 500.0)), atol=1e-9, label="theta = 500")
