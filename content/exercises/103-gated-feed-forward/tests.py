"""Checks for the gated feed-forward exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))


@check("silu gives 0 at 0, 0.7311 at 1, and -0.2689 at -1")
def test_silu_values(sub):
    silu = has(sub, "silu")
    out = silu(np.array([0.0, 1.0, -1.0, 3.0, -3.0]))
    shape(out, (5,), label="silu output")
    close(out[:3], np.array([0.0, 0.7310585786, -0.2689414214]), atol=1e-8, label="silu at 0, 1, -1")
    close(out[3:], np.array([2.857722380, -0.142277620]), atol=1e-7, label="silu at 3, -3")
    close(silu(np.zeros((2, 3))), np.zeros((2, 3)), atol=1e-12, label="shape (2, 3)")


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


@check("silu(z) - silu(-z) equals z, and silu is the smooth ReLU: the slope at 0 is 0.5")
def test_silu_identity(sub):
    silu = has(sub, "silu")
    rng = np.random.default_rng(0)
    z = rng.normal(size=50) * 3
    close(silu(z) - silu(-z), z, atol=1e-10, label="silu(z) - silu(-z)")
    h = 1e-6
    slope = (silu(np.array([h]))[0] - silu(np.array([-h]))[0]) / (2 * h)
    close(slope, 0.5, atol=1e-6, label="slope of silu at 0")
    zz = np.linspace(-6, 6, 241)
    close(silu(zz), zz * sigmoid_ref(zz), atol=1e-12, label="against z * sigmoid(z)")


@check("swiglu reproduces the two-unit hand example")
def test_swiglu_hand(sub):
    swiglu = has(sub, "swiglu")
    W_gate = np.array([[-3.0, 1.0], [0.0, 1.0]])
    W_up = np.eye(2)
    W_down = np.array([[1.0, 0.0], [1.0, -1.0]])
    out = swiglu(np.array([1.0, 2.0]), W_gate, W_up, W_down)
    shape(out, (2,), label="output for one token")
    close(out, np.array([5.5731671, -5.7154448]), atol=1e-5, label="x = [1, 2]")
    out2 = swiglu(np.array([-1.0, 2.0]), W_gate, W_up, W_down)
    hidden2 = np.array([-2.8577224, 1.4621172])
    close(out2, hidden2 @ W_down, atol=1e-5, label="x = [-1, 2]")


@check("swiglu matches a direct computation, with SiLU on the gate only, for a batch of tokens")
def test_swiglu_direct(sub):
    swiglu = has(sub, "swiglu")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 6, 5))
    W_gate = rng.normal(size=(5, 7)) * 0.5
    W_up = rng.normal(size=(5, 7)) * 0.5
    W_down = rng.normal(size=(7, 5)) * 0.5
    out = swiglu(x, W_gate, W_up, W_down)
    shape(out, (2, 6, 5), label="swiglu output on a (2, 6, 5) batch")
    gate = x @ W_gate
    want = (gate * sigmoid_ref(gate) * (x @ W_up)) @ W_down
    close(out, want, atol=1e-9, label="direct computation")
    close(swiglu(x, W_gate, 2.0 * W_up, W_down), 2.0 * out, atol=1e-9, label="doubling W_up doubles the output")
    close(swiglu(x, np.zeros_like(W_gate), W_up, W_down), np.zeros((2, 6, 5)), atol=1e-12, label="a zero gate closes every unit")


@check("ffn_parameter_count gives 24 and 24 at d=2, d_ff=6, and 128 for the width at d_ff=192")
def test_count_small(sub):
    count = has(sub, "ffn_parameter_count")
    equal(int(count(2, 6, False)), 24, label="plain network, d=2, d_ff=6")
    equal(int(count(2, 6, True)), 24, label="gated network, d=2, d_ff=6")
    equal(int(count(64, 192, True)), 3 * 64 * 128, label="gated network, d=64, d_ff=192 (d_h = 128)")
    equal(int(count(64, 192, False)), 2 * 64 * 192, label="plain network, d=64, d_ff=192")
    if not isinstance(count(2, 6, True), (int, np.integer)):
        fail("ffn_parameter_count should return an int")


@check("At d=64 and d_ff=256 the gated count is 3 * 64 * 171, within 1 percent of the plain count")
def test_count_rounding(sub):
    count = has(sub, "ffn_parameter_count")
    plain = int(count(64, 256, False))
    gated = int(count(64, 256, True))
    equal(plain, 32768, label="plain count")
    equal(gated, 32832, label="gated count (d_h rounds from 170.67 to 171)")
    if abs(gated - plain) / plain > 0.01:
        fail(f"the gated count {gated} should be within 1 percent of the plain count {plain}")
