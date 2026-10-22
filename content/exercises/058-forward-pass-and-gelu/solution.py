"""Reference solution: GELU, its slope, and a forward pass."""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_GELU_CUBIC = 0.044715


# Written in the chapter "Why a stack of layers needs a bend".
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# Written in the chapter "Why a stack of layers needs a bend".
def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def gelu(x: np.ndarray) -> np.ndarray:
    u = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    u = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    t = np.tanh(u)
    du = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_CUBIC * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du


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
