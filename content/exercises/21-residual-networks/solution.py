"""Reference solution: residual blocks, and the gradient they keep alive."""

import numpy as np


def residual_block_forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, tuple]:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h1 = x @ W1.T + b1
    a1 = np.maximum(h1, 0.0)
    h2 = a1 @ W2.T + b2
    y = x + h2
    cache = (x, W1, W2, h1, a1)
    return y, cache


def residual_block_backward(dY: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
    x, W1, W2, h1, a1 = cache

    dh2 = dY
    db2 = dh2.sum(axis=0)
    dW2 = dh2.T @ a1
    da1 = dh2 @ W2
    dh1 = da1 * (h1 > 0.0)
    db1 = dh1.sum(axis=0)
    dW1 = dh1.T @ x

    # The shortcut hands dY straight to dx; the branch adds its own term.
    dx = dY + dh1 @ W1

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return dx, grads


def gradient_norm_through_stack(depth: int, residual: bool) -> float:
    d = 16
    rho = 0.1
    rng = np.random.default_rng(0)

    layers = []
    for _ in range(depth):
        a = rng.normal(size=(d, d))
        q, _ = np.linalg.qr(a)
        layers.append(rho * q)

    g = np.ones(d) / np.sqrt(d)
    for w in reversed(layers):
        g = g + g @ w if residual else g @ w
    return float(np.linalg.norm(g))


def bottleneck_flops(c_in: int, c_mid: int, c_out: int, h: int, w: int) -> int:
    reduce_flops = h * w * c_in * c_mid
    spatial_flops = h * w * 9 * c_mid * c_mid
    expand_flops = h * w * c_mid * c_out
    return int(reduce_flops + spatial_flops + expand_flops)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

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
