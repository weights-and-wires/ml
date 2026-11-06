"""Checks for the position clocks exercise."""

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


@check("offset_rotation(k, d) maps the encoding at p to the encoding at p + k")
def test_offset_rotation(sub):
    rot = has(sub, "offset_rotation")
    d = 12
    ref = reference_encoding(40, d)
    for k in (1, 3, 7):
        R = rot(k, d)
        shape(R, (d, d), label=f"offset_rotation({k}, {d})")
        for p in (0, 5, 20):
            close(R @ ref[p], ref[p + k], atol=1e-9, label=f"R_{k} @ PE({p}) versus PE({p + k})")


@check("offset_rotation is orthogonal, is the identity at k=0, and composes by adding offsets")
def test_offset_rotation_algebra(sub):
    rot = has(sub, "offset_rotation")
    d = 8
    close(rot(0, d), np.eye(d), atol=1e-12, label="offset_rotation(0, 8)")
    R = rot(5, d)
    close(R.T @ R, np.eye(d), atol=1e-10, label="R^T R")
    close(rot(2, d) @ rot(3, d), rot(5, d), atol=1e-10, label="R_2 @ R_3 versus R_5")
    close(rot(-3, d) @ rot(3, d), np.eye(d), atol=1e-10, label="R_-3 @ R_3")


@check("add_positions adds the first T rows, leaves its input alone, and separates identical tokens")
def test_add_positions(sub):
    add = has(sub, "add_positions")
    pe = reference_encoding(10, 6)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    before = x.copy()
    out = add(x, pe)
    shape(out, (4, 6), label="add_positions output")
    close(out, before + pe[:4], atol=1e-12, label="x + pe[:4]")
    if not np.array_equal(x, before):
        fail("add_positions changed its input array; return a new array instead")
    same = np.tile(rng.normal(size=(1, 6)), (4, 1))
    out2 = add(same, pe)
    if np.abs(out2[0] - out2[1]).max() < 1e-3:
        fail("identical tokens at different positions should have different vectors after add_positions")
