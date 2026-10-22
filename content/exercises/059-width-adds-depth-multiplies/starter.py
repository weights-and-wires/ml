"""Count the pieces of a ReLU network and build the sawtooth.

A layer is a dict {"W": array (out, in), "b": array (out,), "activation": name}.
`relu`, `gelu`, and `mlp_forward` are already written, as you wrote them in the
earlier chapters. Fill in the two functions below, in order.
"""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


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


if __name__ == "__main__":
    hinge = [
        {"W": np.array([[1.0]]), "b": np.array([-0.5]), "activation": "relu"},
        {"W": np.array([[1.0]]), "b": np.array([0.0]), "activation": "identity"},
    ]
    print("one hinge, pieces on [0, 1]:", count_linear_regions_1d(hinge, 0.0, 1.0))
    for k in (1, 2, 3, 4):
        net = sawtooth_network(k)
        grid = np.linspace(0.0, 1.0, 11)
        values = [float(mlp_forward(net, np.array([v]))[0]) for v in grid]
        print(f"depth {k}: pieces = {count_linear_regions_1d(net, 0.0, 1.0)}, "
              f"values on a grid of 11 = {np.round(values, 3)}")
