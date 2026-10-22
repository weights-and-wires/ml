"""The forward pass of a residual block, and how far it moves its input."""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Replace every negative entry with 0 and leave the others alone.

    Args:
        z: Array of any shape.

    Returns:
        A new array with the same shape as z. Do not change z in place.
    """
    raise NotImplementedError("relu")


def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    """Forward pass of a residual block y = x + F(x), with no ReLU after the sum.

    The branch F is two affine layers with a ReLU between them:
    h1 = x @ W1.T + b1, a1 = relu(h1), h2 = a1 @ W2.T + b2, y = x + h2.

    Args:
        x: Input, shape (n, d), one example per row.
        params: Dict with keys "W1", "b1", "W2", "b2", shaped (d, d), (d,),
            (d, d), (d,).

    Returns:
        A tuple (y, cache). y has shape (n, d). cache is the tuple
        (x, W1, W2, h1, a1), in that order, for the backward pass.
    """
    raise NotImplementedError("residual_block_forward")


def identity_gap(x: np.ndarray, params: dict, residual: bool) -> float:
    """Largest absolute difference between a block's output and its input.

    When residual is True, the block's output is y = x + F(x). When residual is
    False, the block is a plain block and its output is F(x) alone, which
    equals y - x for the y your residual_block_forward returns.

    Args:
        x: Input, shape (n, d).
        params: Same dict as residual_block_forward.
        residual: Whether the block has the shortcut.

    Returns:
        The maximum over all entries of |output - x|, as a float.
    """
    raise NotImplementedError("identity_gap")


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
