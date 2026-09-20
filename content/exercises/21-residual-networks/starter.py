"""Residual blocks, and the gradient they keep alive at depth.

Everything here runs on the CPU in small sizes: the point is the mechanism,
not the scale. The `__main__` block below is where you can plug in a larger
run on a GPU once the functions are correct.
"""

import numpy as np


def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    """Forward pass of a residual block y = x + F(x).

    The branch F is two affine layers with a ReLU between them:
    h1 = x @ W1.T + b1, a1 = relu(h1), h2 = a1 @ W2.T + b2, y = x + h2.

    Args:
        x: Input, shape (N, d).
        params: Dict with keys "W1", "b1", "W2", "b2", shapes (d, d), (d,),
            (d, d), (d,).

    Returns:
        A tuple (y, cache) where y has shape (N, d) and cache holds whatever
        residual_block_backward needs to compute the gradients.
    """
    raise NotImplementedError("residual_block_forward")


def residual_block_backward(dY: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
    """Backward pass of residual_block_forward.

    Args:
        dY: Upstream gradient, shape (N, d), matching the block's output.
        cache: The second element returned by residual_block_forward.

    Returns:
        A tuple (dx, grads) where dx has shape (N, d) and grads is a dict with
        keys "W1", "b1", "W2", "b2" holding the gradients with the same shapes
        as the corresponding entries of params.
    """
    raise NotImplementedError("residual_block_backward")


def gradient_norm_through_stack(depth: int, residual: bool) -> float:
    """Backpropagate a fixed unit gradient through a stack of linear layers.

    Build `depth` layers, each with weight matrix W_l = rho * Q_l for a fixed
    rho and a random orthogonal Q_l, so every layer's Jacobian has operator
    norm exactly rho. When residual is True, layer l computes
    x_l = x_{l-1} + W_l @ x_{l-1}; when False, it computes x_l = W_l @ x_{l-1}.
    Backpropagate a fixed unit-norm vector from the output through every layer
    down to the input, and return the norm of what arrives.

    Seed every random draw so the result is the same on every call.

    Args:
        depth: Number of layers in the stack.
        residual: Whether each layer adds its input (residual) or not (plain).

    Returns:
        The Euclidean norm of the gradient that reaches the input.
    """
    raise NotImplementedError("gradient_norm_through_stack")


def bottleneck_flops(c_in: int, c_mid: int, c_out: int, h: int, w: int) -> int:
    """Multiply-add count of a bottleneck block's three convolutions.

    The block is a 1x1 convolution from c_in to c_mid channels, a 3x3
    convolution from c_mid to c_mid channels, and a 1x1 convolution from
    c_mid to c_out channels, all at spatial size h by w. The shortcut is not
    counted here.

    Args:
        c_in: Input channel count.
        c_mid: Bottleneck (reduced) channel count.
        c_out: Output channel count.
        h: Output height.
        w: Output width.

    Returns:
        The total number of multiply-add operations, as an int.
    """
    raise NotImplementedError("bottleneck_flops")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # A small residual network: `depth` blocks of width `d`, trained by plain
    # gradient descent on a synthetic nonlinear regression target. This is
    # quick on a CPU at this size and trivial on a GPU at a larger one --
    # raise `depth`, `d`, and `n` once your functions pass every check.
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
        residual_error = pred - y
        loss = float(np.mean(residual_error**2))

        if step % 50 == 0:
            print(f"step {step:4d}: loss {loss:.4f}")

        d_pred = 2.0 * residual_error / n
        d_w_out = d_pred.T @ h
        d_b_out = d_pred.sum(axis=0)
        d_h = d_pred @ w_out

        for block, cache in zip(reversed(blocks), reversed(caches)):
            d_h, grads = residual_block_backward(d_h, cache)
            for key in block:
                block[key] -= lr * grads[key]

        w_out -= lr * d_w_out
        b_out -= lr * d_b_out

    print(f"\nbottleneck flops (256 -> 64 -> 256 at 14x14): "
          f"{bottleneck_flops(256, 64, 256, 14, 14):,}")
    for depth_probe in (10, 50):
        r = gradient_norm_through_stack(depth_probe, residual=True)
        p = gradient_norm_through_stack(depth_probe, residual=False)
        print(f"depth {depth_probe:>3}: residual grad norm {r:.3e}, plain grad norm {p:.3e}")
