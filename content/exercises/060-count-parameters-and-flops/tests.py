"""Checks for the exercise on counting a network's cost."""

import numpy as np

from mlcheck import check, equal, fail, has


def _net(dims, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]


@check("parameter_count gives 139 for a 4-8-8-3 network")
def test_params_hand(sub):
    parameter_count = has(sub, "parameter_count")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    # (4*8 + 8) + (8*8 + 8) + (8*3 + 3) = 40 + 72 + 27
    equal(int(parameter_count(net)), 139, label="parameters of 4-8-8-3")
    wide = _net([4, 16, 16, 3], ["relu", "relu", "identity"])
    equal(int(parameter_count(wide)), 403, label="parameters of 4-16-16-3")


@check("The sawtooth network has 6k + 1 parameters")
def test_params_sawtooth(sub):
    parameter_count = has(sub, "parameter_count")
    sawtooth_network = has(sub, "sawtooth_network")
    for k in range(1, 8):
        equal(int(parameter_count(sawtooth_network(k))), 6 * k + 1, label=f"parameters at depth {k}")


@check("flop_count gives 275 for a 4-8-8-3 network with ReLU, ReLU, identity")
def test_flops_hand(sub):
    flop_count = has(sub, "flop_count")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    # 80 + 144 + 51
    equal(int(flop_count(net)), 275, label="flops of 4-8-8-3")


@check("flop_count charges 9 per output for GELU and 0 for identity")
def test_flops_gelu(sub):
    flop_count = has(sub, "flop_count")
    net = _net([4, 8, 3], ["gelu", "identity"])
    # layer 1: 2*4*8 + 8 + 9*8 = 144; layer 2: 2*8*3 + 3 + 0 = 51
    equal(int(flop_count(net)), 195, label="flops of a 4-8-3 network with a GELU layer")


@check("A layer that is not square catches a swapped shape")
def test_flops_rect(sub):
    flop_count = has(sub, "flop_count")
    net = _net([3, 10], ["relu"])
    # 2*3*10 + 10 + 1*10 = 80, a swapped shape would give 2*3*10 + 3 + 3 = 66
    equal(int(flop_count(net)), 80, label="flops of one 3-to-10 ReLU layer")


@check("flop_count adds up over stacked layers")
def test_flops_additive(sub):
    flop_count = has(sub, "flop_count")
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


@check("A one-hidden-layer network with 2**k units has 3 * 2**k + 1 parameters")
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
