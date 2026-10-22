"""Checks for the exercise on counting pieces and folding."""

import numpy as np

from mlcheck import check, close, equal, fail, has


def _one_hidden(kinks, out_weights, in_weight=1.0):
    n = len(kinks)
    return [
        {"W": np.full((n, 1), in_weight), "b": -np.asarray(kinks, dtype=float) * in_weight, "activation": "relu"},
        {"W": np.asarray(out_weights, dtype=float).reshape(1, n), "b": np.array([0.0]), "activation": "identity"},
    ]


def _tent(x):
    return 2.0 * np.minimum(x, 1.0 - x)


@check("A single hinge splits [0, 1] into 2 pieces")
def test_one_hinge(sub):
    count = has(sub, "count_linear_regions_1d")
    got = count(_one_hidden([0.5], [1.0]), 0.0, 1.0)
    equal(int(got), 2, label="pieces of relu(x - 0.5)")


@check("A network with no ReLU has 1 piece, and so does a hinge whose bend lies outside the interval")
def test_one_piece(sub):
    count = has(sub, "count_linear_regions_1d")
    linear = [
        {"W": np.array([[2.0], [-1.0]]), "b": np.array([1.0, 0.5]), "activation": "identity"},
        {"W": np.array([[3.0, 1.0]]), "b": np.array([4.0]), "activation": "identity"},
    ]
    equal(int(count(linear, 0.0, 1.0)), 1, label="identity-only network")
    equal(int(count(_one_hidden([5.0], [1.0]), 0.0, 1.0)), 1, label="relu(x - 5) on [0, 1], always cut to 0")
    equal(int(count(_one_hidden([-5.0], [1.0]), 0.0, 1.0)), 1, label="relu(x + 5) on [0, 1], always passes")


@check("A crossing that changes nothing does not count (the merge step)")
def test_merge(sub):
    count = has(sub, "count_linear_regions_1d")
    net = _one_hidden([0.3, 0.7], [1.0, 0.0])
    equal(int(count(net, 0.0, 1.0)), 2, label="the unit with output weight 0 adds nothing")
    equal(int(count(_one_hidden([0.5], [0.0]), 0.0, 1.0)), 1, label="a hinge whose output weight is 0")


@check("A one-hidden-layer network with kinks at 5 points has 6 pieces")
def test_five_kinks(sub):
    count = has(sub, "count_linear_regions_1d")
    net = _one_hidden([0.1, 0.3, 0.5, 0.7, 0.9], [1.0, -2.0, 1.5, 1.0, -1.0])
    equal(int(count(net, 0.0, 1.0)), 6, label="5 hinges, 6 pieces")
    # the same hinges seen on a narrower interval lose the ones outside it
    equal(int(count(net, 0.0, 0.4)), 3, label="the interval [0, 0.4] holds two kinks")


@check("sawtooth_network(k) has k + 1 layers, with k ReLU layers of width 2")
def test_sawtooth_shape(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    for depth in (1, 2, 4):
        net = sawtooth_network(depth)
        equal(len(net), depth + 1, label=f"layers at depth {depth}")
        for layer in net[:-1]:
            equal(layer["W"].shape[0], 2, label="hidden width")
            equal(layer["activation"], "relu", label="hidden activation")
        equal(net[-1]["W"].shape, (1, 2), label="output W shape")
        equal(net[-1]["activation"], "identity", label="output activation")
        equal(net[0]["W"].shape, (2, 1), label="first W shape")


@check("sawtooth_network(k) matches the tent map applied k times")
def test_sawtooth_closed_form(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    mlp_forward = has(sub, "mlp_forward")
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.0, 1.0, size=30)
    for depth in (1, 2, 3, 4, 5):
        net = sawtooth_network(depth)
        got = np.array([mlp_forward(net, np.array([x]))[0] for x in xs])
        want = xs.copy()
        for _ in range(depth):
            want = _tent(want)
        close(got, want, atol=1e-10, label=f"depth {depth}")


@check("sawtooth_network(k) has exactly 2**k pieces on [0, 1]")
def test_sawtooth_pieces(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    count = has(sub, "count_linear_regions_1d")
    for depth in (1, 2, 3, 4, 5):
        got = int(count(sawtooth_network(depth), 0.0, 1.0))
        if got != 2**depth:
            fail(f"depth {depth}: got {got} pieces, expected {2**depth}")
