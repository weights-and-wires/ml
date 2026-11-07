"""Checks for the signal-size and initialisation exercise."""

import numpy as np

from mlcheck import almost, check, fail, has, shape


@check("layer_gain returns n_in * scale**2")
def test_layer_gain(sub):
    layer_gain = has(sub, "layer_gain")
    almost(layer_gain(2, 1.0), 2.0, tol=1e-12, label="two inputs, scale 1")
    almost(layer_gain(128, 0.1), 1.28, tol=1e-12, label="128 inputs, scale 0.1")
    almost(layer_gain(10000, 0.01), 1.0, tol=1e-12, label="10000 inputs, scale 0.01")


@check("depth_gain raises the gain to the power of the depth")
def test_depth_gain(sub):
    depth_gain = has(sub, "depth_gain")
    almost(depth_gain(0.7, 50), 0.7**50, tol=1e-20, label="0.7 over 50 layers")
    almost(depth_gain(1.4, 50), 1.4**50, tol=1.0, label="1.4 over 50 layers")
    almost(depth_gain(1.0, 50), 1.0, tol=1e-12, label="gain 1 stays 1")
    almost(depth_gain(3.0, 0), 1.0, tol=1e-12, label="zero layers change nothing")


@check("forward_variance returns depth + 1 values, and one linear layer multiplies the mean square by n_in * scale**2")
def test_forward_variance_layer(sub):
    forward_variance = has(sub, "forward_variance")
    layer_gain = has(sub, "layer_gain")
    out = forward_variance(5, [64] * 6, 0.1, "linear", num_trials=300, seed=0)
    shape(out, (6,), label="forward_variance output")
    almost(out[0], 1.0, tol=0.05, label="mean square of a standard normal input")
    out0 = forward_variance(0, [64], 0.1, "linear", num_trials=300, seed=0)
    shape(out0, (1,), label="forward_variance with zero layers")

    n, scale = 300, 0.05
    out = forward_variance(1, [n, n], scale, "linear", num_trials=3000, seed=0)
    predicted = layer_gain(n, scale) * out[0]
    almost(out[1], predicted, tol=0.1 * predicted, label="mean square after one layer")


@check("Stacks with gain 0.8 and gain 1.3 shrink and grow the signal at the predicted rate")
def test_stacks(sub):
    forward_variance = has(sub, "forward_variance")
    layer_gain = has(sub, "layer_gain")
    depth_gain = has(sub, "depth_gain")
    n, depth = 256, 10
    for target in (0.8, 1.3):
        scale = np.sqrt(target / n)
        out = forward_variance(depth, [n] * (depth + 1), scale, "linear", num_trials=300, seed=0)
        predicted = depth_gain(layer_gain(n, scale), depth)
        ratio = out[-1] / out[0] / predicted
        if not 0.5 < ratio < 2.0:
            fail(
                f"after {depth} layers with gain {target} the mean square should change to about "
                f"{predicted:.3f} times its start, but it changed to {out[-1] / out[0]:.3f} times"
            )


@check("xavier_scale returns sqrt(2 / (n_in + n_out)) and averages the two gains to 1")
def test_xavier_scale(sub):
    xavier_scale = has(sub, "xavier_scale")
    almost(xavier_scale(100, 100), np.sqrt(2 / 200), tol=1e-10, label="square layer")
    almost(xavier_scale(256, 64), np.sqrt(2 / 320), tol=1e-10, label="256 in, 64 out")
    s = xavier_scale(256, 64)
    almost((256 * s**2 + 64 * s**2) / 2, 1.0, tol=1e-9, label="average of forward and backward gain")


@check("relu_mean_square_ratio is one half for symmetric numbers, at any scale")
def test_ratio(sub):
    ratio = has(sub, "relu_mean_square_ratio")
    almost(ratio(np.array([-2.0, -1.0, 1.0, 2.0])), 0.5, tol=1e-12, label="-2, -1, 1, 2")
    almost(ratio(np.array([1.0, 2.0, 3.0])), 1.0, tol=1e-12, label="all positive")
    almost(ratio(np.array([-1.0, -4.0])), 0.0, tol=1e-12, label="all negative")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 500))
    almost(ratio(x), 0.5, tol=0.01, label="standard normal")
    almost(ratio(7.0 * x), 0.5, tol=0.01, label="scaled by 7")


@check("he_scale returns sqrt(2 / n_in), and a ReLU layer then has gain 1")
def test_he_scale(sub):
    he_scale = has(sub, "he_scale")
    layer_gain = has(sub, "layer_gain")
    almost(he_scale(128), 0.125, tol=1e-10, label="n_in=128")
    almost(he_scale(512), np.sqrt(2 / 512), tol=1e-10, label="n_in=512")
    almost(0.5 * layer_gain(300, he_scale(300)), 1.0, tol=1e-9, label="half of the gain, after ReLU")


@check("With He scale a 50-layer ReLU network keeps its signal size, and with Xavier scale it loses about half per layer")
def test_relu_depth(sub):
    forward_variance = has(sub, "forward_variance")
    he_scale = has(sub, "he_scale")
    xavier_scale = has(sub, "xavier_scale")
    width, depth = 128, 50
    sizes = forward_variance(depth, [width] * (depth + 1), he_scale(width), "relu", num_trials=300, seed=0)
    factor = float((sizes[-1] / sizes[0]) ** (1.0 / depth))
    almost(factor, 1.0, tol=0.1, label="He's per-layer factor")
    if not 0.05 < sizes[-1] / sizes[0] < 20:
        fail(f"after {depth} layers the mean square moved from {sizes[0]:.3f} to {sizes[-1]:.3g}; expected it to stay near 1")
    sizes = forward_variance(
        depth, [width] * (depth + 1), xavier_scale(width, width), "relu", num_trials=300, seed=0
    )
    factor = float((sizes[-1] / sizes[0]) ** (1.0 / depth))
    almost(factor, 0.5, tol=0.05, label="Xavier's per-layer factor through ReLU")
