"""Reference solution for the initialisation and normalisation exercise."""

import numpy as np


def xavier_scale(n_in: int, n_out: int) -> float:
    return float(np.sqrt(2.0 / (n_in + n_out)))


def he_scale(n_in: int) -> float:
    return float(np.sqrt(2.0 / n_in))


def forward_variance(
    depth: int,
    widths: list[int],
    scale: float,
    activation: str,
    num_trials: int = 300,
    seed: int = 0,
) -> np.ndarray:
    if activation not in ("linear", "relu"):
        raise ValueError(f"unknown activation {activation!r}")
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(num_trials, widths[0]))
    variances = [float(np.mean(h**2))]
    for l in range(depth):
        n_in, n_out = widths[l], widths[l + 1]
        w = rng.normal(scale=scale, size=(n_in, n_out))
        pre = h @ w
        h = np.maximum(pre, 0.0) if activation == "relu" else pre
        variances.append(float(np.mean(h**2)))
    return np.array(variances)


def _normalise(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float, axis: int):
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    istd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * istd
    out = gamma * xhat + beta
    cache = (xhat, istd, gamma, axis)
    return out, cache


def _normalise_backward(dout: np.ndarray, cache):
    xhat, istd, gamma, axis = cache
    m = dout.shape[axis]
    # gamma and beta are always per-feature, shape (D,), so their gradients
    # always sum over the batch axis (0), regardless of which axis the
    # forward pass normalised over.
    dbeta = dout.sum(axis=0)
    dgamma = (dout * xhat).sum(axis=0)
    dxhat = dout * gamma
    dx = (istd / m) * (
        m * dxhat
        - dxhat.sum(axis=axis, keepdims=True)
        - xhat * (dxhat * xhat).sum(axis=axis, keepdims=True)
    )
    return dx, dgamma, dbeta


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    return _normalise(x, gamma, beta, eps, axis=0)


def batchnorm_backward(dout: np.ndarray, cache):
    return _normalise_backward(dout, cache)


def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    return _normalise(x, gamma, beta, eps, axis=1)


def layernorm_backward(dout: np.ndarray, cache):
    return _normalise_backward(dout, cache)


if __name__ == "__main__":
    width = 128
    depth = 50

    he = he_scale(width)
    xavier = xavier_scale(width, width)
    var_he = forward_variance(depth, [width] * (depth + 1), he, "relu")
    var_xavier = forward_variance(depth, [width] * (depth + 1), xavier, "relu")
    print(f"He:     variance at layer 0 = {var_he[0]:.3f}, at layer {depth} = {var_he[-1]:.3e}")
    print(f"Xavier: variance at layer 0 = {var_xavier[0]:.3f}, at layer {depth} = {var_xavier[-1]:.3e}")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 8))
    gamma, beta = np.ones(8), np.zeros(8)

    bn_out, _ = batchnorm_forward(x, gamma, beta)
    ln_out, _ = layernorm_forward(x, gamma, beta)
    print(f"batchnorm per-feature mean (should be ~0): {bn_out.mean(axis=0)[:3]}")
    print(f"layernorm per-example mean (should be ~0): {ln_out.mean(axis=1)[:3]}")
