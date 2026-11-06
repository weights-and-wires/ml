"""Checks for the signal-size exercise."""

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


@check("forward_variance returns depth + 1 values and starts at the input's mean square")
def test_forward_variance_shape(sub):
    forward_variance = has(sub, "forward_variance")
    out = forward_variance(5, [64] * 6, 0.1, "linear", num_trials=300, seed=0)
    shape(out, (6,), label="forward_variance output")
    almost(out[0], 1.0, tol=0.05, label="mean square of a standard normal input")
    out0 = forward_variance(0, [64], 0.1, "linear", num_trials=300, seed=0)
    shape(out0, (1,), label="forward_variance with zero layers")


@check("one linear layer multiplies the mean square by n_in * scale**2")
def test_forward_variance_single_layer(sub):
    forward_variance = has(sub, "forward_variance")
    layer_gain = has(sub, "layer_gain")
    n, scale = 300, 0.05
    out = forward_variance(1, [n, n], scale, "linear", num_trials=3000, seed=0)
    predicted = layer_gain(n, scale) * out[0]
    almost(out[1], predicted, tol=0.1 * predicted, label="mean square after one layer")


@check("a stack with gain below 1 shrinks the signal at the predicted rate")
def test_shrinking_stack(sub):
    forward_variance = has(sub, "forward_variance")
    layer_gain = has(sub, "layer_gain")
    depth_gain = has(sub, "depth_gain")
    n, depth = 256, 10
    scale = np.sqrt(0.8 / n)
    out = forward_variance(depth, [n] * (depth + 1), scale, "linear", num_trials=300, seed=0)
    predicted = depth_gain(layer_gain(n, scale), depth)
    ratio = out[-1] / out[0] / predicted
    if not 0.5 < ratio < 2.0:
        fail(
            f"after {depth} layers with gain 0.8 the mean square should fall to about "
            f"{predicted:.3f} times its start, but it fell to {out[-1] / out[0]:.3f} times"
        )


@check("a stack with gain above 1 grows the signal at the predicted rate")
def test_growing_stack(sub):
    forward_variance = has(sub, "forward_variance")
    layer_gain = has(sub, "layer_gain")
    depth_gain = has(sub, "depth_gain")
    n, depth = 256, 10
    scale = np.sqrt(1.3 / n)
    out = forward_variance(depth, [n] * (depth + 1), scale, "linear", num_trials=300, seed=0)
    predicted = depth_gain(layer_gain(n, scale), depth)
    ratio = out[-1] / out[0] / predicted
    if not 0.5 < ratio < 2.0:
        fail(
            f"after {depth} layers with gain 1.3 the mean square should grow to about "
            f"{predicted:.1f} times its start, but it grew to {out[-1] / out[0]:.1f} times"
        )
