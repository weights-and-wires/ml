"""Reference solution for the multilayer perceptrons exercise."""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_GELU_CUBIC = 0.044715

# Flops charged per output element for each activation, used by flop_count.
# A ReLU costs one comparison-and-select. The tanh-approximate GELU costs the
# nine elementwise arithmetic operations the formula below performs, counting
# the tanh call itself as a single flop. identity costs nothing.
_ACTIVATION_FLOPS = {"relu": 1, "gelu": 9, "identity": 0}


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def gelu(x: np.ndarray) -> np.ndarray:
    inner = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    t = np.tanh(inner)
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_CUBIC * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * d_inner


def _apply_activation(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return relu(z)
    if name == "gelu":
        return gelu(z)
    if name == "identity":
        return z
    raise ValueError(f"unknown activation {name!r}")


def mlp_forward(params: list[dict], x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=float)
    for layer in params:
        z = h @ layer["W"].T + layer["b"]
        h = _apply_activation(layer["activation"], z)
    return h


def sawtooth_network(depth: int) -> list[dict]:
    # The tent map T(x) = 2 relu(x) - 4 relu(x - 1/2) maps [0, 1] onto [0, 1]
    # two-to-one. Stacking `depth` copies computes T composed with itself
    # `depth` times, which has 2**depth linear pieces on [0, 1]. Each hidden
    # layer feeds the next tent map's two probes, x and x - 1/2, so the affine
    # map folding in the previous layer's combination (2 h1 - 4 h2) is fused
    # into the next layer's weight matrix.
    params = []
    # First layer: probes x and x - 1/2 directly from the scalar input.
    params.append({
        "W": np.array([[1.0], [1.0]]),
        "b": np.array([0.0, -0.5]),
        "activation": "relu",
    })
    for _ in range(depth - 1):
        # Combine the previous tent output y = 2 h1 - 4 h2, then probe y and
        # y - 1/2 for the next tent map.
        params.append({
            "W": np.array([[2.0, -4.0], [2.0, -4.0]]),
            "b": np.array([0.0, -0.5]),
            "activation": "relu",
        })
    # Output layer: combine the last tent's two ReLU units, no nonlinearity.
    params.append({
        "W": np.array([[2.0, -4.0]]),
        "b": np.array([0.0]),
        "activation": "identity",
    })
    return params


def count_linear_regions_1d(params: list[dict], x_min: float, x_max: float) -> int:
    for layer in params[:-1]:
        if layer["activation"] not in ("relu", "identity"):
            raise ValueError("count_linear_regions_1d only supports relu/identity layers")
    if params[-1]["activation"] != "identity":
        raise ValueError("the output layer must be linear for a piece count to make sense")
    if params[0]["W"].shape[1] != 1:
        raise ValueError("count_linear_regions_1d is defined for a scalar input")

    # segments: list of (left, right, slopes, intercepts) where slopes and
    # intercepts describe, on (left, right), every unit of the *input to the
    # next layer* as an exactly affine function of x: value = slope * x + intercept.
    segments = [(x_min, x_max, np.array([1.0]), np.array([0.0]))]

    for layer_index, layer in enumerate(params):
        W, b = layer["W"], layer["b"]
        is_output = layer_index == len(params) - 1
        new_segments = []
        for left, right, slopes, intercepts in segments:
            pre_slopes = W @ slopes
            pre_intercepts = W @ intercepts + b

            crossings = []
            for s, c in zip(pre_slopes, pre_intercepts):
                if abs(s) > 1e-12:
                    root = -c / s
                    if left + 1e-9 < root < right - 1e-9:
                        crossings.append(root)
            bounds = [left] + sorted(set(crossings)) + [right]

            for a, bnd in zip(bounds[:-1], bounds[1:]):
                mid = 0.5 * (a + bnd)
                if is_output:
                    post_slopes, post_intercepts = pre_slopes, pre_intercepts
                else:
                    value = pre_slopes * mid + pre_intercepts
                    mask = value > 0.0
                    post_slopes = np.where(mask, pre_slopes, 0.0)
                    post_intercepts = np.where(mask, pre_intercepts, 0.0)
                new_segments.append((a, bnd, post_slopes, post_intercepts))
        segments = new_segments

    # Merge adjacent segments whose final scalar affine coefficients agree,
    # so a crossing that turned out not to change the output does not count.
    merged = [segments[0]]
    for left, right, slopes, intercepts in segments[1:]:
        prev_left, prev_right, prev_slopes, prev_intercepts = merged[-1]
        if np.allclose(slopes, prev_slopes, atol=1e-8) and np.allclose(
            intercepts, prev_intercepts, atol=1e-8
        ):
            merged[-1] = (prev_left, right, prev_slopes, prev_intercepts)
        else:
            merged.append((left, right, slopes, intercepts))
    return len(merged)


def parameter_count(params: list[dict]) -> int:
    return sum(layer["W"].size + layer["b"].size for layer in params)


def flop_count(params: list[dict]) -> int:
    total = 0
    for layer in params:
        out_dim, in_dim = layer["W"].shape
        total += 2 * in_dim * out_dim  # multiply-accumulate for the matmul
        total += out_dim  # bias add
        total += _ACTIVATION_FLOPS[layer["activation"]] * out_dim
    return total


if __name__ == "__main__":
    x = np.linspace(0.0, 1.0, 9)
    print("relu(x - 0.5):", relu(x - 0.5))
    print("gelu(x - 0.5):", np.round(gelu(x - 0.5), 4))

    net = sawtooth_network(depth=3)
    grid = np.linspace(0.0, 1.0, 17)
    values = np.array([mlp_forward(net, np.array([v]))[0] for v in grid])
    print("sawtooth(depth=3) on a grid:", np.round(values, 4))
    print("linear regions on [0, 1]:", count_linear_regions_1d(net, 0.0, 1.0))
    print("parameter count:", parameter_count(net), "flop count:", flop_count(net))
