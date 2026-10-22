"""GELU, its slope, and a forward pass.

A layer is a dict {"W": array (out, in), "b": array (out,), "activation": name}
with name one of "relu", "gelu", or "identity". A network is a list of layers.
`relu` and `relu_grad` are already written, as you wrote them in the last chapter.
Fill in the three functions below, in order.
"""

import numpy as np


# Written in the chapter "Why a stack of layers needs a bend".
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# Written in the chapter "Why a stack of layers needs a bend".
def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


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


if __name__ == "__main__":
    params = [
        {"W": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
         "b": np.array([0.0, 0.0, -1.0]), "activation": "relu"},
        {"W": np.array([[1.0, 1.0, 1.0]]), "b": np.array([0.5]), "activation": "identity"},
    ]
    x = np.array([2.0, -3.0])
    print("pre-activations of layer 1:", x @ params[0]["W"].T + params[0]["b"])
    print("network output:", mlp_forward(params, x))
    print("gelu(1), gelu(-1):", gelu(np.array([1.0, -1.0])))
    print("gelu_grad(1):", gelu_grad(np.array([1.0])))
