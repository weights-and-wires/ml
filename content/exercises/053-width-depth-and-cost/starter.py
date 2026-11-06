"""Count the pieces of a folded network, then count its cost.

A layer is a dict {"W": array (out, in), "b": array (out,), "activation": name}.
`relu`, `gelu`, and `mlp_forward` are already written, as you wrote them in the
earlier chapters. Fill in the five functions below, in order.
"""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)

# Flops charged per output entry for each bend. ReLU is one comparison. GELU
# counts the nine arithmetic operations of its tanh formula, with the tanh
# call as one flop. Identity does nothing. These are counting conventions.
_BEND_FLOPS = {"relu": 1, "gelu": 9, "identity": 0}


# Written in the chapter "Why a stack of layers needs a bend".
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# Written in the chapter "Run a network forward, with a sharp or smooth bend".
def gelu(x: np.ndarray) -> np.ndarray:
    u = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


# Written in the chapter "Run a network forward, with a sharp or smooth bend".
def mlp_forward(params: list[dict], x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=float)
    for layer in params:
        z = h @ layer["W"].T + layer["b"]
        name = layer["activation"]
        if name == "relu":
            h = relu(z)
        elif name == "gelu":
            h = gelu(z)
        elif name == "identity":
            h = z
        else:
            raise ValueError(f"unknown activation {name!r}")
    return h


def count_linear_regions_1d(params: list[dict], x_min: float, x_max: float) -> int:
    """Count the straight pieces of a one-input ReLU network on [x_min, x_max].

    Args:
        params: Layers with "relu" or "identity" activations, the first with
            W of shape (width, 1).
        x_min: Left end of the interval.
        x_max: Right end of the interval.

    Returns:
        The number of pieces of the output as an int. A crossing that does not
        change the output does not count.
    """
    raise NotImplementedError("count_linear_regions_1d")


def sawtooth_network(depth: int) -> list[dict]:
    """Build the network that computes the tent map `depth` times on [0, 1].

    Args:
        depth: Number of folds, at least 1.

    Returns:
        depth + 1 layers: depth ReLU layers of width 2, then one identity layer
        with a single output.
    """
    raise NotImplementedError("sawtooth_network")


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
    hinge = [
        {"W": np.array([[1.0]]), "b": np.array([-0.5]), "activation": "relu"},
        {"W": np.array([[1.0]]), "b": np.array([0.0]), "activation": "identity"},
    ]
    print("one hinge, pieces on [0, 1]:", count_linear_regions_1d(hinge, 0.0, 1.0))
    for k in (1, 2, 3, 4):
        print(f"depth {k}: pieces = {count_linear_regions_1d(sawtooth_network(k), 0.0, 1.0)}")
    net = _make_net([4, 8, 8, 3], ["relu", "relu", "identity"])
    print("4-8-8-3: parameters", parameter_count(net), "flops", flop_count(net),
          "stored activations", activation_memory(net, 4))
    for k in (1, 2, 5, 10):
        print(f"sawtooth depth {k}: {parameter_count(sawtooth_network(k))} parameters "
              f"(6k + 1 = {6 * k + 1}); one hidden layer would need {3 * 2**k + 1}")
