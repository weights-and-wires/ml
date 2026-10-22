"""The backward pass of a residual block.

relu and residual_block_forward are finished already. You write relu_mask,
branch_gradient, and residual_block_backward.
"""

import numpy as np


# Written in the chapter "Why deeper networks got worse, and the residual block".
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


# Written in the chapter "Why deeper networks got worse, and the residual block".
def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h1 = x @ W1.T + b1
    a1 = relu(h1)
    h2 = a1 @ W2.T + b2
    y = x + h2
    cache = (x, W1, W2, h1, a1)
    return y, cache


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


if __name__ == "__main__":
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
