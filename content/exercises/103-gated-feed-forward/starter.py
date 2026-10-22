"""SiLU, the SwiGLU feed-forward network, and its weight budget."""

import numpy as np


def silu(z: np.ndarray) -> np.ndarray:
    """Return z * sigmoid(z), entry by entry.

    It must not overflow or warn for large negative z such as -800.

    Args:
        z: Array of any shape.

    Returns:
        Array with the same shape as z.
    """
    raise NotImplementedError("silu")


def swiglu(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    """Apply a SwiGLU feed-forward network to every row of x.

    swiglu(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    SiLU goes on the gate branch only.

    Args:
        x: Array of shape (..., d_model).
        W_gate: Array of shape (d_model, d_h).
        W_up: Array of shape (d_model, d_h).
        W_down: Array of shape (d_h, d_model).

    Returns:
        Array of shape (..., d_model).
    """
    raise NotImplementedError("swiglu")


def ffn_parameter_count(d_model: int, d_ff: int, gated: bool) -> int:
    """Return the number of weights in a feed-forward network, ignoring biases.

    A plain network has two matrices, d_model by d_ff and d_ff by d_model.
    A gated network has three matrices of the same size, with the hidden width
    d_h = round(2 * d_ff / 3) in place of d_ff.

    Args:
        d_model: Width of a token vector.
        d_ff: The feed-forward width of the plain network.
        gated: True for the SwiGLU network, False for the plain one.

    Returns:
        The weight count as an int.
    """
    raise NotImplementedError("ffn_parameter_count")


if __name__ == "__main__":
    print("silu at -3, 0, 1, 3:", np.round(silu(np.array([-3.0, 0.0, 1.0, 3.0])), 4))

    x = np.array([1.0, 2.0])
    W_gate = np.array([[-3.0, 1.0], [0.0, 1.0]])
    W_up = np.eye(2)
    W_down = np.array([[1.0, 0.0], [1.0, -1.0]])
    print("swiglu on the hand example:", np.round(swiglu(x, W_gate, W_up, W_down), 3))

    print("plain weights at d=64, d_ff=256:", ffn_parameter_count(64, 256, gated=False))
    print("gated weights at d=64, d_ff=256:", ffn_parameter_count(64, 256, gated=True))
