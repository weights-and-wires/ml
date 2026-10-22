"""Parameters, flops, and stored activations of a network.

A layer is a dict {"W": array (out, in), "b": array (out,), "activation": name}
with name one of "relu", "gelu", or "identity". `sawtooth_network` is already
written, as you wrote it in the last chapter. Fill in the three functions below,
in order.
"""

import numpy as np

# Flops charged per output entry for each bend. ReLU is one comparison. GELU
# counts the nine arithmetic operations of its tanh formula, with the tanh
# call as one flop. Identity does nothing. These are counting conventions.
_BEND_FLOPS = {"relu": 1, "gelu": 9, "identity": 0}


# Written in the chapter "Width adds bends and depth multiplies them".
def sawtooth_network(depth: int) -> list[dict]:
    layers = [{"W": np.array([[1.0], [1.0]]), "b": np.array([0.0, -0.5]), "activation": "relu"}]
    for _ in range(depth - 1):
        layers.append(
            {"W": np.array([[2.0, -4.0], [2.0, -4.0]]), "b": np.array([0.0, -0.5]), "activation": "relu"}
        )
    layers.append({"W": np.array([[2.0, -4.0]]), "b": np.array([0.0]), "activation": "identity"})
    return layers


def parameter_count(params: list[dict]) -> int:
    """Return the total number of parameters.

    Args:
        params: A list of layers.

    Returns:
        The number of entries in every W and every b, as an int.
    """
    raise NotImplementedError("parameter_count")


def flop_count(params: list[dict]) -> int:
    """Return the flops of one forward pass on one input.

    Args:
        params: A list of layers.

    Returns:
        The sum over layers of 2 * in * out for the product, out for the bias,
        and out times the bend cost (relu 1, gelu 9, identity 0), as an int.
    """
    raise NotImplementedError("flop_count")


def activation_memory(params: list[dict], in_dim: int) -> int:
    """Return how many floats a forward pass holds for one input.

    Args:
        params: A list of layers.
        in_dim: The length of the input vector.

    Returns:
        The input size plus the output size of every layer, as an int.
    """
    raise NotImplementedError("activation_memory")


def _make_net(dims, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]


if __name__ == "__main__":
    net = _make_net([4, 8, 8, 3], ["relu", "relu", "identity"])
    print("4-8-8-3: parameters", parameter_count(net), "flops", flop_count(net),
          "stored activations", activation_memory(net, 4))
    wide = _make_net([4, 16, 16, 3], ["relu", "relu", "identity"])
    print("4-16-16-3: parameters", parameter_count(wide), "flops", flop_count(wide))
    for k in (1, 2, 5, 10):
        print(f"sawtooth depth {k}: {parameter_count(sawtooth_network(k))} parameters "
              f"(6k + 1 = {6 * k + 1}); one hidden layer would need {3 * 2**k + 1}")
