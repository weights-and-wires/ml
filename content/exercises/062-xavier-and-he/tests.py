"""Checks for the Xavier and He exercise."""

import numpy as np

from mlcheck import almost, check, fail, has


@check("xavier_scale returns sqrt(2 / (n_in + n_out)) and averages the two gains to 1")
def test_xavier_scale(sub):
    xavier_scale = has(sub, "xavier_scale")
    almost(xavier_scale(100, 100), np.sqrt(2 / 200), tol=1e-10, label="square layer")
    almost(xavier_scale(256, 64), np.sqrt(2 / 320), tol=1e-10, label="256 in, 64 out")
    s = xavier_scale(256, 64)
    almost((256 * s**2 + 64 * s**2) / 2, 1.0, tol=1e-9, label="average of forward and backward gain")


@check("relu_mean_square_ratio is exactly one half for symmetric numbers")
def test_ratio_by_hand(sub):
    ratio = has(sub, "relu_mean_square_ratio")
    almost(ratio(np.array([-2.0, -1.0, 1.0, 2.0])), 0.5, tol=1e-12, label="-2, -1, 1, 2")
    almost(ratio(np.array([1.0, 2.0, 3.0])), 1.0, tol=1e-12, label="all positive")
    almost(ratio(np.array([-1.0, -4.0])), 0.0, tol=1e-12, label="all negative")


@check("relu_mean_square_ratio is about one half for random zero-mean numbers, at any scale")
def test_ratio_random(sub):
    ratio = has(sub, "relu_mean_square_ratio")
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


@check("with He scale, a 50-layer ReLU network keeps its signal size")
def test_he_depth(sub):
    forward_variance = has(sub, "forward_variance")
    he_scale = has(sub, "he_scale")
    width, depth = 128, 50
    sizes = forward_variance(depth, [width] * (depth + 1), he_scale(width), "relu", num_trials=300, seed=0)
    factor = float((sizes[-1] / sizes[0]) ** (1.0 / depth))
    almost(factor, 1.0, tol=0.1, label="He's per-layer factor")
    if not 0.05 < sizes[-1] / sizes[0] < 20:
        fail(f"after {depth} layers the mean square moved from {sizes[0]:.3f} to {sizes[-1]:.3g}; expected it to stay near 1")


@check("with Xavier scale, a 50-layer ReLU network loses about half the size per layer")
def test_xavier_depth(sub):
    forward_variance = has(sub, "forward_variance")
    xavier_scale = has(sub, "xavier_scale")
    width, depth = 128, 50
    sizes = forward_variance(
        depth, [width] * (depth + 1), xavier_scale(width, width), "relu", num_trials=300, seed=0
    )
    factor = float((sizes[-1] / sizes[0]) ** (1.0 / depth))
    almost(factor, 0.5, tol=0.05, label="Xavier's per-layer factor through ReLU")
