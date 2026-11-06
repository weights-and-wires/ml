"""Reference solution: the residual block, forward and backward."""

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


def relu_mask(h: np.ndarray) -> np.ndarray:
    return (h > 0).astype(float)


def branch_gradient(dY: np.ndarray, cache: tuple) -> np.ndarray:
    x, W1, W2, h1, a1 = cache
    da1 = dY @ W2
    dh1 = da1 * relu_mask(h1)
    return dh1 @ W1


def residual_block_backward(dY: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
    x, W1, W2, h1, a1 = cache

    dh2 = dY
    db2 = dh2.sum(axis=0)
    dW2 = dh2.T @ a1
    da1 = dh2 @ W2
    dh1 = da1 * relu_mask(h1)
    db1 = dh1.sum(axis=0)
    dW1 = dh1.T @ x

    # The shortcut hands dY straight to dx; the branch adds its own term.
    dx = dY + branch_gradient(dY, cache)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return dx, grads


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
