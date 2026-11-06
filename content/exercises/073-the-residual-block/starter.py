"""The residual block: its forward pass, how far it moves its input, and its backward pass.

You write six functions, in the order the checks test them: `relu`,
`residual_block_forward`, `identity_gap`, `relu_mask`, `branch_gradient`, and
`residual_block_backward`.
"""

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


def relu_mask(h: np.ndarray) -> np.ndarray:
    """The ReLU's switch: 1.0 where h > 0, and 0.0 everywhere else.

    Args:
        h: Array of any shape, the input the ReLU saw on the forward pass.

    Returns:
        A float array with the same shape as h.
    """
    raise NotImplementedError("relu_mask")


def branch_gradient(dY: np.ndarray, cache: tuple) -> np.ndarray:
    """The branch's share of the gradient at the block's input.

    This is dh1 @ W1, where dh1 is the gradient at the first layer's output
    after the ReLU switch. It does not include the shortcut's share, dY.

    Args:
        dY: Upstream gradient, shape (n, d).
        cache: The tuple (x, W1, W2, h1, a1) from residual_block_forward.

    Returns:
        An array of shape (n, d).
    """
    raise NotImplementedError("branch_gradient")


def residual_block_backward(dY: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
    """Backward pass of residual_block_forward.

    Args:
        dY: Upstream gradient, shape (n, d), matching the block's output.
        cache: The second element returned by residual_block_forward.

    Returns:
        A tuple (dx, grads). dx has shape (n, d) and equals dY plus the
        branch's share. grads has keys "W1", "b1", "W2", "b2", each shaped
        like the matching entry of params.
    """
    raise NotImplementedError("residual_block_backward")


def _demo_forward():
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


def _demo_training():
    rng = np.random.default_rng(0)

    # A stack of residual blocks, trained by gradient descent on a regression task.
    d, depth, n = 32, 20, 512
    scale = 0.05 / np.sqrt(d)
    blocks = [
        {
            "W1": rng.normal(size=(d, d)) * scale,
            "b1": np.zeros(d),
            "W2": rng.normal(size=(d, d)) * scale,
            "b2": np.zeros(d),
        }
        for _ in range(depth)
    ]
    w_out = rng.normal(size=(1, d)) * 0.1
    b_out = np.zeros(1)

    x = rng.normal(size=(n, d))
    y = (
        np.sin(2.0 * x[:, 0])
        + 0.5 * x[:, 1] ** 2
        - x[:, 2]
        + 0.05 * rng.normal(size=n)
    ).reshape(-1, 1)

    lr = 0.02
    for step in range(401):
        h = x
        caches = []
        for block in blocks:
            h, cache = residual_block_forward(h, block)
            caches.append(cache)
        pred = h @ w_out.T + b_out
        error = pred - y
        loss = float(np.mean(error**2))

        if step % 50 == 0:
            print(f"step {step:4d}: loss {loss:.4f}")

        d_pred = 2.0 * error / n
        d_w_out = d_pred.T @ h
        d_b_out = d_pred.sum(axis=0)
        d_h = d_pred @ w_out

        for block, cache in zip(reversed(blocks), reversed(caches)):
            d_h, grads = residual_block_backward(d_h, cache)
            for key in block:
                block[key] -= lr * grads[key]

        w_out -= lr * d_w_out
        b_out -= lr * d_b_out


if __name__ == "__main__":
    _demo_forward()
    print()
    _demo_training()
