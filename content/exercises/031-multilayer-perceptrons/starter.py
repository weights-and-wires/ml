"""A multilayer perceptron: bends, a forward pass, folded pieces, and cost.

A layer is a dict {"W": array (out, in), "b": array (out,), "activation": name}
with name one of "relu", "gelu", or "identity". A network is a list of layers.
Fill in the twelve functions below, in order. The lesson walks through each one.
"""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)

# Flops charged per output entry for each bend. ReLU is one comparison. GELU
# counts the nine arithmetic operations of its tanh formula, with the tanh
# call as one flop. Identity does nothing. These are counting conventions.
_BEND_FLOPS = {"relu": 1, "gelu": 9, "identity": 0}


def relu(x: np.ndarray) -> np.ndarray:
    """Return the elementwise ReLU, max(x, 0).

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape.
    """
    raise NotImplementedError("relu")


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Return the elementwise slope of ReLU.

    Args:
        x: An array of any shape.

    Returns:
        A float array of the same shape, 1.0 where x > 0 and 0.0 where x < 0.
        The value at x == 0 does not matter for the checks.
    """
    raise NotImplementedError("relu_grad")


def compose_affine(W1, b1, W2, b2):
    """Merge two affine layers into one.

    Args:
        W1: First layer weights, shape (mid, in).
        b1: First layer constants, shape (mid,).
        W2: Second layer weights, shape (out, mid).
        b2: Second layer constants, shape (out,).

    Returns:
        A pair (W, b) with W of shape (out, in) and b of shape (out,), such that
        W @ x + b equals W2 @ (W1 @ x + b1) + b2 for every x.
    """
    raise NotImplementedError("compose_affine")


def gelu(x: np.ndarray) -> np.ndarray:
    """Return the tanh approximation of GELU.

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape: 0.5 * x * (1 + tanh(u)), with
        u = sqrt(2 / pi) * (x + 0.044715 * x**3).
    """
    raise NotImplementedError("gelu")


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Return the elementwise slope of gelu.

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape. Checked against finite differences.
    """
    raise NotImplementedError("gelu_grad")


def mlp_forward(params: list[dict], x: np.ndarray) -> np.ndarray:
    """Run one input through a list of layers.

    Args:
        params: Layers, each a dict with "W" of shape (out, in), "b" of shape
            (out,), and "activation" in {"relu", "gelu", "identity"}.
        x: One input vector of shape (in,).

    Returns:
        The output vector of the last layer. "identity" leaves the
        pre-activation unchanged. An unknown name raises ValueError.
    """
    raise NotImplementedError("mlp_forward")


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


def hinge_interpolant(xs: np.ndarray, ys: np.ndarray) -> list[dict]:
    """Return a one-hidden-layer ReLU network through every knot (xs[j], ys[j]).

    Use one hinge relu(x - xs[j]) per segment, at the segment's left knot. The
    output weight of a hinge is the CHANGE in slope there (the first hinge's is
    the first slope), and the output bias is ys[0].

    Args:
        xs: Strictly increasing knot positions, shape (n,), with n >= 2.
        ys: The heights at the knots, shape (n,).

    Returns:
        A list of two layers in the format mlp_forward reads: a "relu" layer
        with W of shape (n - 1, 1), then an "identity" layer with W of shape
        (1, n - 1). The network equals the straight-segment interpolant of the
        knots on [xs[0], xs[-1]].

    Raises:
        ValueError: If xs is not strictly increasing or the lengths differ.
    """
    raise NotImplementedError("hinge_interpolant")


def _make_net(dims, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]


if __name__ == "__main__":
    # XOR: lamp on when exactly one switch is up.
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    W_hidden = np.array([[1.0, -1.0], [-1.0, 1.0]])  # rows: x1 - x2 and x2 - x1
    hidden = relu(points @ W_hidden.T)
    print("hidden features:\n", hidden)
    print("h1 + h2 - 0.5:", hidden.sum(axis=1) - 0.5)

    W, b = compose_affine(
        np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0]),
        np.array([[3.0, 1.0]]), np.array([4.0]),
    )
    print("merged W:", W, "merged b:", b)
    print("gelu(1), gelu(-1):", gelu(np.array([1.0, -1.0])))
    print("gelu_grad(1):", gelu_grad(np.array([1.0])))

    params = [
        {"W": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
         "b": np.array([0.0, 0.0, -1.0]), "activation": "relu"},
        {"W": np.array([[1.0, 1.0, 1.0]]), "b": np.array([0.5]), "activation": "identity"},
    ]
    print("network output:", mlp_forward(params, np.array([2.0, -3.0])))

    for k in (1, 2, 3, 4):
        print(f"depth {k}: pieces = {count_linear_regions_1d(sawtooth_network(k), 0.0, 1.0)}")
    net = _make_net([4, 8, 8, 3], ["relu", "relu", "identity"])
    print("4-8-8-3: parameters", parameter_count(net), "flops", flop_count(net),
          "stored activations", activation_memory(net, 4))
