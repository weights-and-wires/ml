"""Checks for the transformer block exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def build_params(rng, d_model, d_h, heads, zero_wo=False, zero_wdown=False):
    def small(*size):
        return rng.normal(size=size) * 0.02

    return {
        "Wq": small(d_model, d_model),
        "Wk": small(d_model, d_model),
        "Wv": small(d_model, d_model),
        "Wo": np.zeros((d_model, d_model)) if zero_wo else small(d_model, d_model),
        "ln1_gain": np.ones(d_model),
        "ln1_bias": np.zeros(d_model),
        "ln2_gain": np.ones(d_model),
        "ln2_bias": np.zeros(d_model),
        "W_gate": small(d_model, d_h),
        "W_up": small(d_model, d_h),
        "W_down": np.zeros((d_h, d_model)) if zero_wdown else small(d_h, d_model),
        "heads": heads,
        "theta": 10000.0,
    }


@check("sinusoidal_encoding is exactly [sin, cos] at every position")
def test_sinusoidal_basic(sub):
    got = has(sub, "sinusoidal_encoding")
    enc = got(T=4, d=8)
    shape(enc, (4, 8), label="sinusoidal_encoding output")
    # Position 0: every angle is zero, so evens are sin(0) = 0 and odds cos(0) = 1.
    close(enc[0, 0::2], np.zeros(4), atol=1e-12, label="row 0 sin columns")
    close(enc[0, 1::2], np.ones(4), atol=1e-12, label="row 0 cos columns")
    # Column 0 (i = 0, omega = 1) at position p should be sin(p).
    close(enc[:, 0], np.sin(np.arange(4, dtype=float)), atol=1e-10, label="column 0")
    close(enc[:, 1], np.cos(np.arange(4, dtype=float)), atol=1e-10, label="column 1")


@check("The sinusoidal encoding at offset k is a fixed linear map of the encoding at p")
def test_sinusoidal_linear_offset(sub):
    got = has(sub, "sinusoidal_encoding")
    d, k, T = 16, 5, 40
    enc = got(T=T, d=d)

    # Build the block-diagonal rotation M_k directly from the closed form, with
    # no reference to the submission's internals.
    i = np.arange(d // 2, dtype=float)
    omega = 10000.0 ** (-2.0 * i / d)
    M = np.zeros((d, d))
    for idx, w in enumerate(omega):
        c, s = np.cos(k * w), np.sin(k * w)
        M[2 * idx, 2 * idx] = c
        M[2 * idx, 2 * idx + 1] = s
        M[2 * idx + 1, 2 * idx] = -s
        M[2 * idx + 1, 2 * idx + 1] = c

    for p in (0, 7, 13, 22):
        predicted = M @ enc[p]
        close(predicted, enc[p + k], atol=1e-8, label=f"offset map at p={p}")


@check("rope_frequencies is a geometric sequence theta ** (-2i/d)")
def test_rope_frequencies(sub):
    got = has(sub, "rope_frequencies")
    d = 8
    freqs = got(d, theta=100.0)
    shape(freqs, (4,), label="rope_frequencies output")
    want = 100.0 ** (-2.0 * np.arange(4) / d)
    close(freqs, want, atol=1e-10, label="rope_frequencies values")


@check("apply_rope preserves vector norms")
def test_rope_norm_preserved(sub):
    apply_rope = has(sub, "apply_rope")
    freqs = has(sub, "rope_frequencies")(8)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 8))
    positions = np.arange(5.0)[:, None]
    angles = positions * freqs[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)

    rotated = apply_rope(x, cos, sin)
    shape(rotated, x.shape, label="apply_rope output")
    close(
        np.linalg.norm(rotated, axis=-1),
        np.linalg.norm(x, axis=-1),
        atol=1e-8,
        label="row norms",
    )


@check("apply_rope makes the query-key score a function of m - n alone")
def test_rope_relative_offset(sub):
    apply_rope = has(sub, "apply_rope")
    rope_frequencies = has(sub, "rope_frequencies")
    d = 16
    freqs = rope_frequencies(d)
    rng = np.random.default_rng(2)
    q = rng.normal(size=d)
    k = rng.normal(size=d)

    def rotated_score(m, n):
        angles_q = m * freqs
        cos_q = np.concatenate([np.cos(angles_q), np.cos(angles_q)])
        sin_q = np.concatenate([np.sin(angles_q), np.sin(angles_q)])
        angles_k = n * freqs
        cos_k = np.concatenate([np.cos(angles_k), np.cos(angles_k)])
        sin_k = np.concatenate([np.sin(angles_k), np.sin(angles_k)])
        q_rot = apply_rope(q, cos_q, sin_q)
        k_rot = apply_rope(k, cos_k, sin_k)
        return float(q_rot @ k_rot)

    offset = 3
    scores = [rotated_score(m, m + offset) for m in (0, 4, 11, 20)]
    spread = max(scores) - min(scores)
    if spread > 1e-6:
        fail(
            "the score at fixed m - n should not depend on the absolute "
            f"position, but it varied by {spread:.3g} across {scores}"
        )
    # And a different offset should, in general, give a different score.
    other = rotated_score(0, 0 + offset + 1)
    if abs(other - scores[0]) < 1e-6:
        fail("changing the offset should change the score, but it did not")


@check("swiglu matches a direct computation of silu-gated matmuls")
def test_swiglu(sub):
    got = has(sub, "swiglu")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 5))
    W_gate = rng.normal(size=(5, 7)) * 0.3
    W_up = rng.normal(size=(5, 7)) * 0.3
    W_down = rng.normal(size=(7, 5)) * 0.3

    out = got(x, W_gate, W_up, W_down)
    shape(out, (6, 5), label="swiglu output")

    gate = x @ W_gate
    silu = gate * (1.0 / (1.0 + np.exp(-gate)))
    want = (silu * (x @ W_up)) @ W_down
    close(out, want, atol=1e-8, label="swiglu value")


@check("transformer_block_forward preserves the input's shape")
def test_block_shape(sub):
    got = has(sub, "transformer_block_forward")
    rng = np.random.default_rng(4)
    T, d_model, heads, d_h = 10, 24, 4, 32
    x = rng.normal(size=(T, d_model)) * 0.1
    params = build_params(rng, d_model, d_h, heads)
    mask = np.triu(np.full((T, T), -1e9), k=1)

    out = got(x, params, mask)
    shape(out, (T, d_model), label="transformer_block_forward output")
    if not np.all(np.isfinite(out)):
        fail("transformer_block_forward produced non-finite values")


@check("Zeroing both output projections makes the block an exact identity")
def test_block_identity(sub):
    got = has(sub, "transformer_block_forward")
    rng = np.random.default_rng(5)
    T, d_model, heads, d_h = 8, 16, 2, 20
    x = rng.normal(size=(T, d_model)) * 0.3
    params = build_params(rng, d_model, d_h, heads, zero_wo=True, zero_wdown=True)
    mask = np.triu(np.full((T, T), -1e9), k=1)

    out = got(x, params, mask)
    close(out, x, atol=1e-10, label="block output with zeroed output projections")


@check("block_parameter_count matches a hand computation")
def test_block_parameter_count(sub):
    got = has(sub, "block_parameter_count")
    d_model, d_ff, heads = 64, 192, 8
    d_h = round(2 * d_ff / 3)  # 128, exactly, since d_ff is a multiple of 3
    want = 4 * d_model**2 + 3 * d_model * d_h + 2 * (2 * d_model)
    equal(int(got(d_model, d_ff, heads)), int(want), label="block_parameter_count")

    d_model2, d_ff2, heads2 = 128, 512, 16
    d_h2 = round(2 * d_ff2 / 3)
    want2 = 4 * d_model2**2 + 3 * d_model2 * d_h2 + 2 * (2 * d_model2)
    equal(int(got(d_model2, d_ff2, heads2)), int(want2), label="second configuration")
