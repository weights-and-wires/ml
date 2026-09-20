"""Activations, a forward pass, and the exact sawtooth network.

A layer is a dict `{"W": ndarray(out, in), "b": ndarray(out,), "activation":
name}`, with `name` one of `"relu"`, `"gelu"`, or `"identity"`. A network is a
list of such layers, applied in order: `h = activation(h @ W.T + b)`.

Fill in the eight functions below.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Return the elementwise ReLU, max(x, 0).

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape.
    """
    raise NotImplementedError("relu")


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Return the elementwise derivative of relu.

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape, 1 where x > 0 and 0 where x < 0. The value
        at x == 0 does not matter for the checks.
    """
    raise NotImplementedError("relu_grad")


def gelu(x: np.ndarray) -> np.ndarray:
    """Return the tanh approximation to GELU.

    GELU(x) = 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))).

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape.
    """
    raise NotImplementedError("gelu")


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Return the elementwise derivative of the tanh approximation to GELU.

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape.
    """
    raise NotImplementedError("gelu_grad")


def mlp_forward(params: list[dict], x: np.ndarray) -> np.ndarray:
    """Run a forward pass through an arbitrary list of layers.

    Args:
        params: A list of layers, each a dict with keys "W" (out, in), "b"
            (out,), and "activation" (one of "relu", "gelu", "identity").
        x: An input vector, shape (in_dim,), where in_dim is the first
            layer's input dimension.

    Returns:
        The network's output, shape (out_dim,), where out_dim is the last
        layer's output dimension.
    """
    raise NotImplementedError("mlp_forward")


def sawtooth_network(depth: int) -> list[dict]:
    """Build a ReLU network computing the depth-k sawtooth on [0, 1] exactly.

    The sawtooth of depth k is the tent map T(x) = 2 min(x, 1 - x), composed
    with itself k times. It has exactly 2**k linear pieces on [0, 1].

    Build the network as `depth` hidden ReLU layers of width 2, followed by a
    final linear (identity) layer that produces the scalar output. Every
    layer's weights should be exact rational numbers, so the network computes
    the sawtooth to machine precision, not merely approximately.

    Args:
        depth: The number of tent-map compositions, k >= 1.

    Returns:
        A list of `depth + 1` layers in the `params` format `mlp_forward`
        expects, taking a 1-dimensional input to a 1-dimensional output.
    """
    raise NotImplementedError("sawtooth_network")


def count_linear_regions_1d(params: list[dict], x_min: float, x_max: float) -> int:
    """Count the exact number of linear pieces of a scalar ReLU network.

    Every hidden layer must use "relu" or "identity", and the output layer
    must use "identity" — the piece count is undefined for a network with a
    smooth nonlinearity, since it is not piecewise linear at all. Propagate
    the breakpoints exactly through the network: track, for each interval
    between breakpoints, the affine formula (slope, intercept) that every
    unit computes there, split an interval wherever a unit's pre-activation
    crosses zero, and merge adjacent intervals whose final output formula
    turns out to agree.

    Args:
        params: A network as in mlp_forward, taking a 1-dimensional input.
        x_min: Left end of the domain to count pieces over.
        x_max: Right end of the domain to count pieces over.

    Returns:
        The number of maximal intervals in (x_min, x_max) on which the
        network's output is an affine function of the input.
    """
    raise NotImplementedError("count_linear_regions_1d")


def parameter_count(params: list[dict]) -> int:
    """Return the total number of scalar parameters in a network.

    Args:
        params: A network as in mlp_forward.

    Returns:
        The sum, over every layer, of the number of entries in W plus the
        number of entries in b.
    """
    raise NotImplementedError("parameter_count")


def flop_count(params: list[dict]) -> int:
    """Return the flop cost of one forward pass through a network.

    Charge 2 flops per multiply-accumulate in the matrix-vector product,
    1 flop per bias addition, 1 flop per output element for a relu
    (a comparison and a select), 9 flops per output element for the tanh
    approximation to gelu (its eight arithmetic operations plus the tanh
    call itself), and 0 for identity.

    Args:
        params: A network as in mlp_forward.

    Returns:
        The total flop count for a single input vector, as an int.
    """
    raise NotImplementedError("flop_count")


if __name__ == "__main__":
    x = np.linspace(0.0, 1.0, 9)
    print("relu(x - 0.5):", relu(x - 0.5))
    print("gelu(x - 0.5):", np.round(gelu(x - 0.5), 4))

    net = sawtooth_network(depth=3)
    grid = np.linspace(0.0, 1.0, 17)
    values = np.array([mlp_forward(net, np.array([v]))[0] for v in grid])
    print("sawtooth(depth=3) on a grid:", np.round(values, 4))
    print("linear regions on [0, 1]:", count_linear_regions_1d(net, 0.0, 1.0))
    print("parameter count:", parameter_count(net), "flop count:", flop_count(net))
