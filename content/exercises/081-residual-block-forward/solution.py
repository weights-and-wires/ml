"""Reference solution: the forward pass of a residual block."""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h1 = x @ W1.T + b1
    a1 = relu(h1)
    h2 = a1 @ W2.T + b2
    y = x + h2
    cache = (x, W1, W2, h1, a1)
    return y, cache


def identity_gap(x: np.ndarray, params: dict, residual: bool) -> float:
    y, _ = residual_block_forward(x, params)
    out = y if residual else y - x  # a plain block outputs F(x) alone
    return float(np.max(np.abs(out - x)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, depth = 32, 20

    blocks = [
        {
            "W1": rng.normal(size=(d, d)) * 0.05,
            "b1": np.zeros(d),
            "W2": rng.normal(size=(d, d)) * 0.05,
            "b2": np.zeros(d),
        }
        for _ in range(depth)
    ]
    x = rng.normal(size=(1, d))

    h_res, h_plain = x, x
    for block in blocks:
        h_res, _ = residual_block_forward(h_res, block)
        y, _ = residual_block_forward(h_plain, block)
        h_plain = y - h_plain  # the plain block's output, F(h)

    print(f"input norm:                 {np.linalg.norm(x):.4f}")
    print(f"after {depth} residual blocks:   {np.linalg.norm(h_res):.4f}")
    print(f"after {depth} plain blocks:      {np.linalg.norm(h_plain):.2e}")

    x2 = np.array([[1.0, 2.0], [-1.0, 3.0]])
    p2 = {
        "W1": np.eye(2),
        "b1": np.array([0.0, -1.0]),
        "W2": np.array([[2.0, 0.0], [0.0, 0.5]]),
        "b2": np.array([0.1, 0.0]),
    }
    y2, _ = residual_block_forward(x2, p2)
    print("\nthe worked example from the chapter:")
    print(y2)
