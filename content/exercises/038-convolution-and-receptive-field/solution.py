"""Reference solution for the convolution and receptive-field exercise."""

import numpy as np


def conv2d_single(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    k = w.shape[0]
    h_out = x.shape[0] - k + 1
    w_out = x.shape[1] - k + 1
    y = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            y[i, j] = b + np.sum(w * x[i : i + k, j : j + k])
    return y


def conv2d_naive(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    n, c_in, h, w_dim = x.shape
    c_out, _, k, _ = w.shape
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w_dim + 2 * padding - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    out = np.zeros((n, c_out, h_out, w_out))
    for img in range(n):
        for o in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    top, left = i * stride, j * stride
                    patch = xp[img, :, top : top + k, left : left + k]
                    out[img, o, i, j] = b[o] + np.sum(w[o] * patch)
    return out


def param_counts(n_in: int, n_out: int, c_in: int, c_out: int, k: int) -> dict:
    return {
        "dense": n_in * n_out + n_out,
        "conv": c_out * c_in * k * k + c_out,
        "separable": c_in * k * k + c_in * c_out + c_out,
    }


def conv_output_shape(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    reach = dilation * (kernel_size - 1) + 1
    return (size + 2 * padding - reach) // stride + 1


def receptive_field_per_layer(layers: list[tuple[int, int]]) -> list[int]:
    fields = []
    r = 1
    jump = 1
    for kernel_size, stride in layers:
        r = r + (kernel_size - 1) * jump
        jump = jump * stride
        fields.append(r)
    return fields


def max_pool_forward(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    n, c, h, w = x.shape
    h_out = conv_output_shape(h, pool_size, stride)
    w_out = conv_output_shape(w, pool_size, stride)
    out = np.empty((n, c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        top = i * stride
        for j in range(w_out):
            left = j * stride
            window = x[:, :, top : top + pool_size, left : left + pool_size]
            out[:, :, i, j] = window.max(axis=(2, 3))
    return out


def max_pool_backward(
    dout: np.ndarray, x: np.ndarray, pool_size: int, stride: int
) -> np.ndarray:
    dx = np.zeros_like(x, dtype=float)
    n, c, _, _ = x.shape
    _, _, h_out, w_out = dout.shape
    for image in range(n):
        for channel in range(c):
            for i in range(h_out):
                for j in range(w_out):
                    top, left = i * stride, j * stride
                    window = x[image, channel, top : top + pool_size, left : left + pool_size]
                    index = int(np.argmax(window))
                    row, col = divmod(index, pool_size)
                    dx[image, channel, top + row, left + col] += dout[image, channel, i, j]
    return dx


def average_pool_forward(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    n, c, h, w = x.shape
    h_out = conv_output_shape(h, pool_size, stride)
    w_out = conv_output_shape(w, pool_size, stride)
    out = np.empty((n, c, h_out, w_out), dtype=float)
    for i in range(h_out):
        top = i * stride
        for j in range(w_out):
            left = j * stride
            window = x[:, :, top : top + pool_size, left : left + pool_size]
            out[:, :, i, j] = window.mean(axis=(2, 3))
    return out


def average_pool_backward(
    dout: np.ndarray, x: np.ndarray, pool_size: int, stride: int
) -> np.ndarray:
    dx = np.zeros(x.shape, dtype=float)
    _, _, h_out, w_out = dout.shape
    scale = float(pool_size * pool_size)
    for i in range(h_out):
        top = i * stride
        for j in range(w_out):
            left = j * stride
            dx[:, :, top : top + pool_size, left : left + pool_size] += (
                dout[:, :, i, j][:, :, None, None] / scale
            )
    return dx


if __name__ == "__main__":
    x = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)
    edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    print("vertical-edge response:")
    print(conv2d_single(x, edge, 0.0))

    rng = np.random.default_rng(0)
    imgs = rng.normal(size=(2, 3, 8, 8))
    kernels = rng.normal(size=(4, 3, 3, 3)) * 0.2
    out = conv2d_naive(imgs, kernels, np.zeros(4), stride=2, padding=1)
    print("batched output shape:", out.shape)

    counts = param_counts(224 * 224 * 3, 1000, 3, 64, 3)
    print("photo layer, dense vs conv:", counts["dense"], "vs", counts["conv"])
    counts = param_counts(0, 0, 64, 64, 3)
    print("64 to 64 channels, conv vs separable:", counts["conv"], "vs", counts["separable"])

    stack = [(3, 2, 1), (3, 2, 1), (3, 1, 1)]
    size = 32
    for k, s, p in stack:
        size = conv_output_shape(size, k, s, p)
        print(f"after a {k}x{k} layer with stride {s} and padding {p}: {size} by {size}")
    print("receptive field after each layer:",
          receptive_field_per_layer([(k, s) for k, s, _ in stack]))

    grid = np.array([[1, 3, 2, 5], [4, 9, 1, 0], [6, 2, 7, 1], [1, 1, 3, 2]], dtype=float)
    print("2x2 max pool:")
    print(max_pool_forward(grid[None, None], 2, 2)[0, 0])
    dout = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    print("max-pool gradient:")
    print(max_pool_backward(dout, grid[None, None], 2, 2)[0, 0])
    print("2x2 average pool:")
    print(average_pool_forward(grid[None, None], 2, 2)[0, 0])
    print("average-pool gradient:")
    print(average_pool_backward(dout, grid[None, None], 2, 2)[0, 0])
