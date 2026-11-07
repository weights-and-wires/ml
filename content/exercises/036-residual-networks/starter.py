"""A residual block, the gradient through a deep stack, and the cost of a bottleneck.

You write nine functions, in the order the checks test them:
`residual_block_forward`, `identity_gap`, `residual_block_backward`,
`orthogonal_matrix`, `scalar_gradient_factor`, `gradient_norm_through_stack`,
`conv_flops`, `bottleneck_flops`, and `bottleneck_savings`.
"""

import numpy as np



def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    """Forward pass of a residual block y = x + F(x), with no ReLU after the sum.

    The branch F is two affine layers with a ReLU between them:
    h1 = x @ W1.T + b1, a1 = max(h1, 0), h2 = a1 @ W2.T + b2, y = x + h2.

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


def orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthogonal matrix: it rotates or flips a vector and keeps its length.

    Draw a Gaussian matrix, take the Q factor of its QR decomposition, and
    multiply each column of Q by the sign of the matching diagonal entry of R.

    Args:
        d: Size of the square matrix.
        rng: Source of randomness. Draw from it exactly once, with
            rng.normal(size=(d, d)).

    Returns:
        A (d, d) array Q with Q.T @ Q equal to the identity.
    """
    raise NotImplementedError("orthogonal_matrix")


def scalar_gradient_factor(J: float, depth: int, residual: bool) -> float:
    """The toy model: every layer multiplies the gradient by the same number.

    A plain layer multiplies the gradient by J. A residual layer multiplies it
    by 1 + J. Return the factor after depth layers.

    Args:
        J: The slope of one layer's branch.
        depth: Number of layers.
        residual: Whether each layer has a shortcut.

    Returns:
        J ** depth for a plain stack, (1 + J) ** depth for a residual stack.
    """
    raise NotImplementedError("scalar_gradient_factor")


def gradient_norm_through_stack(depth: int, residual: bool) -> float:
    """Length of the gradient that reaches the input of a stack of layers.

    Build `depth` layers of width 16, each 0.1 times orthogonal_matrix(16, rng),
    drawing every matrix from one rng = np.random.default_rng(0), in order.
    Start from g = np.ones(16) / np.sqrt(16). Walk the layers in reverse order.
    A plain layer sends g to g @ W. A residual layer sends g to g + g @ W.

    Args:
        depth: Number of layers in the stack.
        residual: Whether each layer has a shortcut.

    Returns:
        The Euclidean norm of g after the last step, as a float.
    """
    raise NotImplementedError("gradient_norm_through_stack")


def conv_flops(c_in: int, c_out: int, k: int, h: int, w: int) -> int:
    """Multiply-adds of one convolution that keeps the picture at h by w.

    Assume stride 1, a k by k kernel, and no bias. Each of the h * w * c_out
    output numbers needs c_in * k * k products.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        k: Kernel height and width.
        h: Output height.
        w: Output width.

    Returns:
        The multiply-add count, as an int.
    """
    raise NotImplementedError("conv_flops")


def bottleneck_flops(c_in: int, c_mid: int, c_out: int, h: int, w: int) -> int:
    """Multiply-adds of a bottleneck block's three convolutions.

    The block is a 1x1 convolution from c_in to c_mid channels, a 3x3
    convolution from c_mid to c_mid channels, and a 1x1 convolution from c_mid
    to c_out channels, all at h by w. The shortcut is not counted.

    Args:
        c_in: Input channels.
        c_mid: Channels inside the block.
        c_out: Output channels.
        h: Height.
        w: Width.

    Returns:
        The total multiply-add count, as an int.
    """
    raise NotImplementedError("bottleneck_flops")


def bottleneck_savings(c: int, h: int, w: int) -> float:
    """How many times cheaper a bottleneck is than two full-width 3x3 convolutions.

    The bottleneck goes c -> c // 4 -> c. The alternative is two 3x3
    convolutions from c channels to c channels.

    Args:
        c: Outer channel count, divisible by 4.
        h: Height.
        w: Width.

    Returns:
        The cost of the two 3x3 convolutions divided by the bottleneck's cost,
        as a float.
    """
    raise NotImplementedError("bottleneck_savings")


if __name__ == "__main__":
    # The worked example: two examples with two features each.
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    params = {
        "W1": np.eye(2),
        "b1": np.array([0.0, -1.0]),
        "W2": np.array([[2.0, 0.0], [0.0, 0.5]]),
        "b2": np.array([0.1, 0.0]),
    }
    y, cache = residual_block_forward(x, params)
    print("block output:")
    print(y)
    dx, grads = residual_block_backward(np.ones((2, 2)), cache)
    print("dx:")
    print(dx)

    print("\ndepth   residual length   plain length")
    for depth in (1, 10, 50):
        r = gradient_norm_through_stack(depth, residual=True)
        p = gradient_norm_through_stack(depth, residual=False)
        print(f"{depth:5d}   {r:15.4f}   {p:12.3e}")

    print("\n256 -> 64 -> 256 at 14x14:")
    print(f"  bottleneck:     {bottleneck_flops(256, 64, 256, 14, 14):,}")
    print(f"  savings:        {bottleneck_savings(256, 14, 14):.2f}x")
