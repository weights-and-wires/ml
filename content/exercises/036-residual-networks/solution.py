"""Reference solution: a residual block, the gradient through a deep stack, and bottleneck cost."""

import numpy as np



def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h1 = x @ W1.T + b1
    a1 = np.maximum(h1, 0.0)
    h2 = a1 @ W2.T + b2
    y = x + h2
    cache = (x, W1, W2, h1, a1)
    return y, cache


def identity_gap(x: np.ndarray, params: dict, residual: bool) -> float:
    y, _ = residual_block_forward(x, params)
    out = y if residual else y - x  # a plain block outputs F(x) alone
    return float(np.max(np.abs(out - x)))


def residual_block_backward(dY: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
    x, W1, W2, h1, a1 = cache

    dh2 = dY
    db2 = dh2.sum(axis=0)
    dW2 = dh2.T @ a1
    da1 = dh2 @ W2
    dh1 = da1 * (h1 > 0)  # the ReLU is a switch
    db1 = dh1.sum(axis=0)
    dW1 = dh1.T @ x

    # The shortcut hands dY straight to dx; the branch adds its own term.
    dx = dY + dh1 @ W1

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return dx, grads


def orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    # QR alone is biased; fixing the signs of the columns makes q evenly random.
    return q * np.sign(np.diag(r))


def scalar_gradient_factor(J: float, depth: int, residual: bool) -> float:
    factor = 1.0 + J if residual else J
    return float(factor**depth)


def gradient_norm_through_stack(depth: int, residual: bool) -> float:
    d = 16
    scale = 0.1
    rng = np.random.default_rng(0)

    layers = [scale * orthogonal_matrix(d, rng) for _ in range(depth)]

    g = np.ones(d) / np.sqrt(d)
    for w in reversed(layers):
        g = g + g @ w if residual else g @ w
    return float(np.linalg.norm(g))


def conv_flops(c_in: int, c_out: int, k: int, h: int, w: int) -> int:
    return int(h * w * c_in * c_out * k * k)


def bottleneck_flops(c_in: int, c_mid: int, c_out: int, h: int, w: int) -> int:
    shrink = conv_flops(c_in, c_mid, 1, h, w)
    work = conv_flops(c_mid, c_mid, 3, h, w)
    grow = conv_flops(c_mid, c_out, 1, h, w)
    return int(shrink + work + grow)


def bottleneck_savings(c: int, h: int, w: int) -> float:
    two_full_width = 2 * conv_flops(c, c, 3, h, w)
    return float(two_full_width / bottleneck_flops(c, c // 4, c, h, w))


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
