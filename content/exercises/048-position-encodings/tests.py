"""Checks for the position encodings exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def reference_encoding(T, d):
    """A direct, loop-based encoding, so the later checks do not lean on the reader's."""
    pe = np.zeros((T, d))
    for p in range(T):
        for i in range(d // 2):
            omega = 10000.0 ** (-2.0 * i / d)
            pe[p, 2 * i] = np.sin(p * omega)
            pe[p, 2 * i + 1] = np.cos(p * omega)
    return pe


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


@check("sinusoidal_encoding gives [0, 1, 0, 1] at position 0 and the hand values at position 1")
def test_encoding_hand_values(sub):
    pe = has(sub, "sinusoidal_encoding")(T=3, d=4)
    shape(pe, (3, 4), label="sinusoidal_encoding output")
    close(pe[0], np.array([0.0, 1.0, 0.0, 1.0]), atol=1e-12, label="position 0")
    want1 = np.array([np.sin(1.0), np.cos(1.0), np.sin(0.01), np.cos(0.01)])
    close(pe[1], want1, atol=1e-10, label="position 1")
    close(pe[1], np.array([0.8415, 0.5403, 0.0100, 0.99995]), atol=1e-4, label="position 1, rounded")


@check("sinusoidal_encoding interleaves sine and cosine and uses speeds 10000 ** (-2i / d)")
def test_encoding_matches_reference(sub):
    pe = has(sub, "sinusoidal_encoding")(T=20, d=16)
    shape(pe, (20, 16), label="sinusoidal_encoding output")
    close(pe, reference_encoding(20, 16), atol=1e-9, label="encoding at T=20, d=16")


@check("The similarity PE @ PE.T has diagonal d/2 and depends only on the distance")
def test_similarity_toeplitz(sub):
    d, T = 16, 24
    pe = has(sub, "sinusoidal_encoding")(T, d)
    G = pe @ pe.T
    close(np.diag(G), np.full(T, d / 2), atol=1e-9, label="diagonal of PE @ PE.T")
    for gap in (1, 3, 7):
        values = np.array([G[p, p + gap] for p in range(T - gap)])
        spread = values.max() - values.min()
        if spread > 1e-9:
            fail(f"similarity at distance {gap} should be constant, but it varied by {spread:.3g}")
    if not G[0, 1] > G[0, 12]:
        fail(f"a neighbour should be more similar than a far position, got {G[0, 1]:.3f} and {G[0, 12]:.3f}")


@check("offset_rotation(k, d) maps the encoding at p to p + k, is orthogonal, and composes by adding offsets")
def test_offset_rotation(sub):
    rot = has(sub, "offset_rotation")
    d = 12
    ref = reference_encoding(40, d)
    for k in (1, 3, 7):
        R = rot(k, d)
        shape(R, (d, d), label=f"offset_rotation({k}, {d})")
        for p in (0, 5, 20):
            close(R @ ref[p], ref[p + k], atol=1e-9, label=f"R_{k} @ PE({p}) versus PE({p + k})")
    d = 8
    close(rot(0, d), np.eye(d), atol=1e-12, label="offset_rotation(0, 8)")
    R = rot(5, d)
    close(R.T @ R, np.eye(d), atol=1e-10, label="R^T R")
    close(rot(2, d) @ rot(3, d), rot(5, d), atol=1e-10, label="R_2 @ R_3 versus R_5")


@check("rope_tables returns (T, d) tables, row 0 is (1, 0), and each speed theta ** (-2j / d) sits in both halves")
def test_tables(sub):
    tables = has(sub, "rope_tables")
    cos, sin = tables(T=6, d=8)
    shape(cos, (6, 8), label="cos table")
    shape(sin, (6, 8), label="sin table")
    close(cos[0], np.ones(8), atol=1e-12, label="cos at position 0")
    close(sin[0], np.zeros(8), atol=1e-12, label="sin at position 0")
    close(cos**2 + sin**2, np.ones((6, 8)), atol=1e-12, label="cos^2 + sin^2")
    d, T, theta = 8, 5, 10000.0
    cos, sin = tables(T, d, theta)
    for j in range(d // 2):
        speed = theta ** (-2.0 * j / d)
        want_cos = np.cos(np.arange(T) * speed)
        want_sin = np.sin(np.arange(T) * speed)
        close(cos[:, j], want_cos, atol=1e-10, label=f"cos column {j}")
        close(cos[:, j + d // 2], want_cos, atol=1e-10, label=f"cos column {j + d // 2}")
        close(sin[:, j], want_sin, atol=1e-10, label=f"sin column {j}")
        close(sin[:, j + d // 2], want_sin, atol=1e-10, label=f"sin column {j + d // 2}")
    cos2, _ = tables(3, 4, 100.0)
    close(cos2[:, 1], np.cos(np.arange(3) * 100.0 ** (-0.5)), atol=1e-10, label="a different theta")


@check("apply_rope turns [1, 0] at position 1 to [cos 1, sin 1], matches an explicit turn, and keeps lengths")
def test_apply_rope(sub):
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
