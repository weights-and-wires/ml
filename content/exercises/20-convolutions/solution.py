"""Reference solution: convolution as a matrix multiply."""

import numpy as np


def conv_output_shape(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    effective_kernel = dilation * (kernel_size - 1) + 1
    return (size + 2 * padding - effective_kernel) // stride + 1


def im2col(x: np.ndarray, kernel_size: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    n, c, h, w_dim = x.shape
    k = kernel_size
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    col = np.zeros((n, c, k, k, h_out, w_out), dtype=x.dtype)
    for dh in range(k):
        h_stop = dh + stride * h_out
        for dw in range(k):
            w_stop = dw + stride * w_out
            col[:, :, dh, dw, :, :] = x_padded[:, :, dh:h_stop:stride, dw:w_stop:stride]
    return col.reshape(n, c * k * k, h_out * w_out)


def col2im(
    dcol: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    n, c, h, w_dim = x_shape
    k = kernel_size
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)

    dcol_r = dcol.reshape(n, c, k, k, h_out, w_out)
    h_padded, w_padded = h + 2 * padding, w_dim + 2 * padding
    dx_padded = np.zeros((n, c, h_padded, w_padded), dtype=dcol.dtype)
    for dh in range(k):
        h_stop = dh + stride * h_out
        for dw in range(k):
            w_stop = dw + stride * w_out
            dx_padded[:, :, dh:h_stop:stride, dw:w_stop:stride] += dcol_r[:, :, dh, dw, :, :]

    if padding == 0:
        return dx_padded
    return dx_padded[:, :, padding:-padding, padding:-padding]


def conv2d_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    n, c_in, h, w_dim = x.shape
    c_out, _, k, _ = w.shape
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)

    col = im2col(x, k, stride, padding)  # (N, C_in*k*k, L)
    w_mat = w.reshape(c_out, -1)  # (C_out, C_in*k*k)
    out = np.einsum("oc,ncl->nol", w_mat, col) + b[None, :, None]
    return out.reshape(n, c_out, h_out, w_out)


def conv2d_backward(
    dY: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, c_in, h, w_dim = x.shape
    c_out, _, k, _ = w.shape

    col = im2col(x, k, stride, padding)  # (N, C_in*k*k, L)
    w_mat = w.reshape(c_out, -1)  # (C_out, C_in*k*k)
    dY_mat = dY.reshape(n, c_out, -1)  # (N, C_out, L)

    db = dY_mat.sum(axis=(0, 2))
    dw_mat = np.einsum("nol,ncl->oc", dY_mat, col)
    dcol = np.einsum("oc,nol->ncl", w_mat, dY_mat)
    dx = col2im(dcol, x.shape, k, stride, padding)

    return dx, dw_mat.reshape(w.shape), db


def max_pool_forward(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    n, c, h, w_dim = x.shape
    h_out = conv_output_shape(h, pool_size, stride, padding=0)
    w_out = conv_output_shape(w_dim, pool_size, stride, padding=0)

    out = np.empty((n, c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        hs = i * stride
        for j in range(w_out):
            ws = j * stride
            window = x[:, :, hs : hs + pool_size, ws : ws + pool_size]
            out[:, :, i, j] = window.max(axis=(2, 3))
    return out


def receptive_field(layers: list[tuple[int, int]]) -> int:
    r = 1
    jump = 1
    for kernel_size, stride in layers:
        r = r + (kernel_size - 1) * jump
        jump *= stride
    return r


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 9, 9))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.1
    b = rng.normal(size=(4,)) * 0.1

    y = conv2d_forward(x, w, b, stride=2, padding=1)
    print(f"conv output shape: {y.shape}")

    dY = rng.normal(size=y.shape)
    dx, dw, db = conv2d_backward(dY, x, w, b, stride=2, padding=1)
    print(f"dx shape: {dx.shape}, dw shape: {dw.shape}, db shape: {db.shape}")

    pooled = max_pool_forward(x, pool_size=3, stride=3)
    print(f"pooled shape: {pooled.shape}")

    stack = [(3, 2), (3, 2), (3, 1)]
    print(f"receptive field of {stack}: {receptive_field(stack)}")
