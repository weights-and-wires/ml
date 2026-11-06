"""Reference solution: slide a filter over a picture, one position at a time."""

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
