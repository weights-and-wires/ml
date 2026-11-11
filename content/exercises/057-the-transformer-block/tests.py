"""Checks for the transformer block exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_params(rng, d_model, d_h, heads, zero_wo=False, zero_wdown=False, theta=10000.0):
    def small(*size):
        return rng.normal(size=size) * 0.3

    return {
        "Wq": small(d_model, d_model),
        "Wk": small(d_model, d_model),
        "Wv": small(d_model, d_model),
        "Wo": np.zeros((d_model, d_model)) if zero_wo else small(d_model, d_model),
        "ln1_gain": 1.0 + 0.2 * rng.normal(size=d_model),
        "ln1_bias": 0.1 * rng.normal(size=d_model),
        "ln2_gain": 1.0 + 0.2 * rng.normal(size=d_model),
        "ln2_bias": 0.1 * rng.normal(size=d_model),
        "W_gate": small(d_model, d_h),
        "W_up": small(d_model, d_h),
        "W_down": np.zeros((d_h, d_model)) if zero_wdown else small(d_h, d_model),
        "heads": heads,
        "theta": theta,
    }


def ref_layer_norm(row, gain, bias):
    mean = row.mean()
    var = ((row - mean) ** 2).mean()
    return (row - mean) / np.sqrt(var + 1e-5) * gain + bias


def ref_rotate(vec, pos, theta):
    """Turn each pair (j, j + h) of one head vector by pos * theta ** (-2j / d)."""
    d = vec.shape[0]
    h = d // 2
    out = vec.copy()
    for j in range(h):
        a = pos * theta ** (-2.0 * j / d)
        out[j] = vec[j] * np.cos(a) - vec[j + h] * np.sin(a)
        out[j + h] = vec[j] * np.sin(a) + vec[j + h] * np.cos(a)
    return out


def ref_block(x, p, mask):
    """A slow, loop-based block, written separately from any array tricks."""
    T, d = x.shape
    heads = p["heads"]
    dh = d // heads
    n1 = np.stack([ref_layer_norm(x[t], p["ln1_gain"], p["ln1_bias"]) for t in range(T)])
    Q, K, V = n1 @ p["Wq"], n1 @ p["Wk"], n1 @ p["Wv"]
    out = np.zeros((T, d))
    for h in range(heads):
        sl = slice(h * dh, (h + 1) * dh)
        q = np.stack([ref_rotate(Q[t, sl], t, p["theta"]) for t in range(T)])
        k = np.stack([ref_rotate(K[t, sl], t, p["theta"]) for t in range(T)])
        for t in range(T):
            scores = np.array([q[t] @ k[s] / np.sqrt(dh) + mask[t, s] for s in range(T)])
            w = np.exp(scores - scores.max())
            w /= w.sum()
            out[t, sl] = sum(w[s] * V[s, sl] for s in range(T))
    x1 = x + out @ p["Wo"]
    n2 = np.stack([ref_layer_norm(x1[t], p["ln2_gain"], p["ln2_bias"]) for t in range(T)])
    g = n2 @ p["W_gate"]
    hidden = g / (1.0 + np.exp(-g)) * (n2 @ p["W_up"])
    return x1 + hidden @ p["W_down"]


def causal_mask(T):
    return np.triu(np.full((T, T), -1e9), k=1)


@check("silu gives 0 at 0, 0.7311 at 1, and -0.2689 at -1, and silu(z) - silu(-z) equals z")
def test_silu_values(sub):
    silu = has(sub, "silu")
    out = silu(np.array([0.0, 1.0, -1.0, 3.0, -3.0]))
    shape(out, (5,), label="silu output")
    close(out[:3], np.array([0.0, 0.7310585786, -0.2689414214]), atol=1e-8, label="silu at 0, 1, -1")
    close(out[3:], np.array([2.857722380, -0.142277620]), atol=1e-7, label="silu at 3, -3")
    close(silu(np.zeros((2, 3))), np.zeros((2, 3)), atol=1e-12, label="shape (2, 3)")
    rng = np.random.default_rng(0)
    z = rng.normal(size=50) * 3
    close(silu(z) - silu(-z), z, atol=1e-10, label="silu(z) - silu(-z)")
    zz = np.linspace(-6, 6, 241)
    close(silu(zz), zz * sigmoid_ref(zz), atol=1e-12, label="against z * sigmoid(z)")


@check("silu is stable: large positive z returns z, large negative z returns a finite value near 0")
def test_silu_stable(sub):
    silu = has(sub, "silu")
    z = np.array([-800.0, -50.0, 50.0, 800.0])
    try:
        with np.errstate(all="raise"):
            out = silu(z)
    except FloatingPointError as err:
        fail(f"silu overflowed or underflowed on {z}: {err}. Write the sigmoid so that np.exp never sees a large positive number")
    if not np.all(np.isfinite(out)):
        fail(f"silu returned non-finite values for {z}: {out}")
    close(out[2:], z[2:], rtol=1e-12, label="large positive z")
    close(out[:2], np.zeros(2), atol=1e-12, label="large negative z")


@check("swiglu reproduces the two-unit hand example and matches a direct computation on a batch")
def test_swiglu(sub):
    swiglu = has(sub, "swiglu")
    W_gate = np.array([[-3.0, 1.0], [0.0, 1.0]])
    W_up = np.eye(2)
    W_down = np.array([[1.0, 0.0], [1.0, -1.0]])
    out = swiglu(np.array([1.0, 2.0]), W_gate, W_up, W_down)
    shape(out, (2,), label="output for one token")
    close(out, np.array([5.5731671, -5.7154448]), atol=1e-5, label="x = [1, 2]")
    out2 = swiglu(np.array([-1.0, 2.0]), W_gate, W_up, W_down)
    close(out2, np.array([-2.8577224, 1.4621172]) @ W_down, atol=1e-5, label="x = [-1, 2]")

    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 6, 5))
    Wg = rng.normal(size=(5, 7)) * 0.5
    Wu = rng.normal(size=(5, 7)) * 0.5
    Wd = rng.normal(size=(7, 5)) * 0.5
    out = swiglu(x, Wg, Wu, Wd)
    shape(out, (2, 6, 5), label="swiglu output on a (2, 6, 5) batch")
    gate = x @ Wg
    want = (gate * sigmoid_ref(gate) * (x @ Wu)) @ Wd
    close(out, want, atol=1e-9, label="direct computation")
    close(swiglu(x, Wg, 2.0 * Wu, Wd), 2.0 * out, atol=1e-9, label="doubling W_up doubles the output")
    close(swiglu(x, np.zeros_like(Wg), Wu, Wd), np.zeros((2, 6, 5)), atol=1e-12, label="a zero gate closes every unit")


@check("ffn_parameter_count gives 24 and 24 at d=2, d_ff=6, and 32832 against 32768 at d=64, d_ff=256")
def test_ffn_count(sub):
    count = has(sub, "ffn_parameter_count")
    equal(int(count(2, 6, False)), 24, label="plain network, d=2, d_ff=6")
    equal(int(count(2, 6, True)), 24, label="gated network, d=2, d_ff=6")
    equal(int(count(64, 192, True)), 3 * 64 * 128, label="gated network, d=64, d_ff=192 (d_h = 128)")
    if not isinstance(count(2, 6, True), (int, np.integer)):
        fail("ffn_parameter_count should return an int")
    equal(int(count(64, 256, False)), 32768, label="plain count")
    equal(int(count(64, 256, True)), 32832, label="gated count (d_h rounds from 170.67 to 171)")


@check("layer_norm turns [1, 2, 3, 6] into [-1.0690, -0.5345, 0, 1.6036]")
def test_layer_norm_hand(sub):
    ln = has(sub, "layer_norm")
    out = ln(np.array([[1.0, 2.0, 3.0, 6.0]]), np.ones(4), np.zeros(4))
    shape(out, (1, 4), label="layer_norm output")
    close(out[0], np.array([-1.0690, -0.5345, 0.0, 1.6036]), atol=1e-4, label="normalised row")


@check("layer_norm gives each row mean 0 and variance 1, ignores overall size, and applies gain and bias last")
def test_layer_norm_properties(sub):
    ln = has(sub, "layer_norm")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 8)) * 3 + 2
    ones, zeros = np.ones(8), np.zeros(8)
    out = ln(x, ones, zeros)
    close(out.mean(axis=-1), np.zeros(5), atol=1e-9, label="row means")
    close(out.var(axis=-1), np.ones(5), atol=1e-4, label="row variances")
    close(ln(10.0 * x, ones, zeros), out, atol=1e-5, label="a token ten times larger")
    gain, bias = rng.normal(size=8), rng.normal(size=8)
    close(ln(x, gain, bias), out * gain + bias, atol=1e-9, label="gain and bias applied after normalising")
    const = ln(np.full((1, 4), 7.0), np.ones(4), np.zeros(4))
    if not np.all(np.isfinite(const)):
        fail("a row of equal entries has variance 0, but layer_norm must stay finite because of eps")
    cube = ln(rng.normal(size=(2, 3, 8)), ones, zeros)
    shape(cube, (2, 3, 8), label="a (2, 3, 8) input")


@check("transformer_block_forward keeps the shape, and zeroing Wo and W_down makes it an exact identity")
def test_block_shape(sub):
    block = has(sub, "transformer_block_forward")
    rng = np.random.default_rng(1)
    T, d, heads, d_h = 8, 16, 4, 20
    x = rng.normal(size=(T, d))
    out = block(x, build_params(rng, d, d_h, heads), causal_mask(T))
    shape(out, (T, d), label="transformer_block_forward output")
    if not np.all(np.isfinite(out)):
        fail("the block produced non-finite values")
    if np.allclose(out, x):
        fail("the block returned its input unchanged, but the sublayers should add something")

    # Zeroed output matrices leave the stream untouched.
    rng = np.random.default_rng(2)
    T, d, heads, d_h = 7, 12, 3, 16
    x = rng.normal(size=(T, d))
    params = build_params(rng, d, d_h, heads, zero_wo=True, zero_wdown=True)
    close(block(x, params, causal_mask(T)), x, atol=1e-10, label="block output with zeroed output matrices")
    params = build_params(rng, d, d_h, heads, zero_wdown=True)
    out = block(x, params, causal_mask(T))
    if np.allclose(out, x):
        fail("with Wo nonzero the attention sublayer should change the stream")


@check("Under a causal mask, changing a later token leaves earlier outputs unchanged")
def test_block_causal(sub):
    block = has(sub, "transformer_block_forward")
    rng = np.random.default_rng(3)
    T, d, heads, d_h = 8, 16, 2, 24
    x = rng.normal(size=(T, d))
    params = build_params(rng, d, d_h, heads)
    x2 = x.copy()
    x2[5:] = rng.normal(size=(T - 5, d))
    a = block(x, params, causal_mask(T))
    b = block(x2, params, causal_mask(T))
    close(a[:5], b[:5], atol=1e-9, label="outputs at positions 0 to 4")
    if np.allclose(a[5:], b[5:]):
        fail("outputs at the changed positions should differ")
    open_a = block(x, params, np.zeros((T, T)))
    open_b = block(x2, params, np.zeros((T, T)))
    if np.allclose(open_a[:5], open_b[:5], atol=1e-6):
        fail("with no mask, earlier positions should see later tokens, so the mask must be what blocks them")


@check("transformer_block_forward matches a slow reference, and the rotary base changes the result")
def test_block_reference(sub):
    block = has(sub, "transformer_block_forward")
    rng = np.random.default_rng(4)
    T, d, heads, d_h = 6, 16, 4, 12
    x = rng.normal(size=(T, d))
    params = build_params(rng, d, d_h, heads)
    mask = causal_mask(T)
    got = block(x, params, mask)
    close(got, ref_block(x, params, mask), atol=1e-8, label="block against the loop-based reference")
    params2 = dict(params)
    params2["theta"] = 50.0
    other = block(x, params2, mask)
    close(other, ref_block(x, params2, mask), atol=1e-8, label="block with theta = 50")
    if np.allclose(got, other, atol=1e-6):
        fail("changing theta should change the output, so rotary embeddings must be applied inside attention")
    no_theta = {k: v for k, v in params.items() if k != "theta"}
    close(block(x, no_theta, mask), got, atol=1e-10, label="a missing theta means 10000")


@check("block_parameter_count gives 41216 for (64, 192, 8) and ignores the number of heads")
def test_parameter_count(sub):
    count = has(sub, "block_parameter_count")
    equal(int(count(64, 192, 8)), 41216, label="d=64, d_ff=192, heads=8")
    d, dff = 128, 512
    d_h = round(2 * dff / 3)
    want = 4 * d**2 + 3 * d * d_h + 4 * d
    equal(int(count(d, dff, 16)), want, label="second configuration")
    equal(int(count(d, dff, 4)), want, label="heads changed, same count")
    try:
        count(64, 192, 5)
    except ValueError:
        return
    fail("heads=5 does not divide d_model=64, so block_parameter_count should raise ValueError")
