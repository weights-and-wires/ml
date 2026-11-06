"""SiLU, the SwiGLU feed-forward network, and its weight budget."""

import numpy as np


def silu(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    sigmoid = 0.5 * (1.0 + np.tanh(z / 2.0))  # equals 1 / (1 + e^-z) without overflow
    return z * sigmoid


def swiglu(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    gate = silu(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down


def ffn_parameter_count(d_model: int, d_ff: int, gated: bool) -> int:
    if not gated:
        return 2 * d_model * d_ff
    d_h = round(2 * d_ff / 3)
    return 3 * d_model * d_h


if __name__ == "__main__":
    print("silu at -3, 0, 1, 3:", np.round(silu(np.array([-3.0, 0.0, 1.0, 3.0])), 4))

    x = np.array([1.0, 2.0])
    W_gate = np.array([[-3.0, 1.0], [0.0, 1.0]])
    W_up = np.eye(2)
    W_down = np.array([[1.0, 0.0], [1.0, -1.0]])
    print("swiglu on the hand example:", np.round(swiglu(x, W_gate, W_up, W_down), 3))

    print("plain weights at d=64, d_ff=256:", ffn_parameter_count(64, 256, gated=False))
    print("gated weights at d=64, d_ff=256:", ffn_parameter_count(64, 256, gated=True))
