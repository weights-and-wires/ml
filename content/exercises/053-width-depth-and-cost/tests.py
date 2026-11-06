"""Checks for the exercise on folding a network and counting its cost."""

import numpy as np

from mlcheck import check, close, equal, fail, has


def _net(dims, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]


def _one_hidden(kinks, out_weights, in_weight=1.0):
    n = len(kinks)
    return [
        {"W": np.full((n, 1), in_weight), "b": -np.asarray(kinks, dtype=float) * in_weight, "activation": "relu"},
        {"W": np.asarray(out_weights, dtype=float).reshape(1, n), "b": np.array([0.0]), "activation": "identity"},
    ]


def _tent(x):
    return 2.0 * np.minimum(x, 1.0 - x)


@check("Piece counts: a hinge splits [0, 1] in 2, while no ReLU or a bend outside the interval gives 1")
def test_pieces_basic(sub):
    count = has(sub, "count_linear_regions_1d")
    equal(int(count(_one_hidden([0.5], [1.0]), 0.0, 1.0)), 2, label="pieces of relu(x - 0.5)")
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


@check("sawtooth_network(k) has the right layer shapes and matches the tent map applied k times")
def test_sawtooth_shape_and_values(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    mlp_forward = has(sub, "mlp_forward")
    for depth in (1, 2, 4):
        net = sawtooth_network(depth)
        equal(len(net), depth + 1, label=f"layers at depth {depth}")
        for layer in net[:-1]:
            equal(layer["W"].shape[0], 2, label="hidden width")
            equal(layer["activation"], "relu", label="hidden activation")
        equal(net[-1]["W"].shape, (1, 2), label="output W shape")
        equal(net[-1]["activation"], "identity", label="output activation")
        equal(net[0]["W"].shape, (2, 1), label="first W shape")
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


@check("parameter_count gives 139 for a 4-8-8-3 network, and 6k + 1 for the sawtooth")
def test_params(sub):
    parameter_count = has(sub, "parameter_count")
    sawtooth_network = has(sub, "sawtooth_network")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    # (4*8 + 8) + (8*8 + 8) + (8*3 + 3) = 40 + 72 + 27
    equal(int(parameter_count(net)), 139, label="parameters of 4-8-8-3")
    wide = _net([4, 16, 16, 3], ["relu", "relu", "identity"])
    equal(int(parameter_count(wide)), 403, label="parameters of 4-16-16-3")
    for k in range(1, 8):
        equal(int(parameter_count(sawtooth_network(k))), 6 * k + 1, label=f"parameters at depth {k}")


@check("flop_count gives 275 for a 4-8-8-3 network and charges 9 per GELU output")
def test_flops(sub):
    flop_count = has(sub, "flop_count")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    # 80 + 144 + 51
    equal(int(flop_count(net)), 275, label="flops of 4-8-8-3")
    gelu_net = _net([4, 8, 3], ["gelu", "identity"])
    # layer 1: 2*4*8 + 8 + 9*8 = 144; layer 2: 2*8*3 + 3 + 0 = 51
    equal(int(flop_count(gelu_net)), 195, label="flops of a 4-8-3 network with a GELU layer")
    # a swapped shape would give 2*3*10 + 3 + 3 = 66
    equal(int(flop_count(_net([3, 10], ["relu"]))), 80, label="flops of one 3-to-10 ReLU layer")
    a = _net([5, 7], ["relu"], seed=1)
    b = _net([7, 4, 2], ["gelu", "identity"], seed=2)
    equal(int(flop_count(a + b)), int(flop_count(a)) + int(flop_count(b)), label="flops of the stack")


@check("activation_memory counts the input and every layer's output")
def test_activation_memory(sub):
    activation_memory = has(sub, "activation_memory")
    sawtooth_network = has(sub, "sawtooth_network")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    equal(int(activation_memory(net, 4)), 23, label="stored activations of 4-8-8-3")
    for k in (1, 3, 6):
        # input 1, then 2 per hidden layer, then 1 output
        equal(int(activation_memory(sawtooth_network(k), 1)), 2 * k + 2, label=f"sawtooth depth {k}")


@check("A one-hidden-layer network with 2**k units has 3 * 2**k + 1 parameters, far more than the sawtooth")
def test_one_layer_cost(sub):
    parameter_count = has(sub, "parameter_count")
    sawtooth_network = has(sub, "sawtooth_network")
    for k in (1, 2, 4, 6):
        n = 2**k
        wide = _net([1, n, 1], ["relu", "identity"])
        got = int(parameter_count(wide))
        if got != 3 * n + 1:
            fail(f"k = {k}: a 1-{n}-1 network has {got} parameters, expected {3 * n + 1}")
    if int(parameter_count(sawtooth_network(10))) >= 3 * 2**10 + 1:
        fail("the depth-10 sawtooth should have far fewer parameters than a one-hidden-layer network")
