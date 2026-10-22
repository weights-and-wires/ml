"""Reference solution: output sizes, receptive fields, and max pooling."""

import numpy as np


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


if __name__ == "__main__":
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
