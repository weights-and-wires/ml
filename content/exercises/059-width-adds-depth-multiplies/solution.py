"""Reference solution: count the pieces of a ReLU network and build the sawtooth."""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


# Written in the chapter "Why a stack of layers needs a bend".
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# Written in the chapter "Run a network forward, with a sharp or smooth bend".
def gelu(x: np.ndarray) -> np.ndarray:
    u = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


# Written in the chapter "Run a network forward, with a sharp or smooth bend".
def mlp_forward(params: list[dict], x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=float)
    for layer in params:
        z = h @ layer["W"].T + layer["b"]
        name = layer["activation"]
        if name == "relu":
            h = relu(z)
        elif name == "gelu":
            h = gelu(z)
        elif name == "identity":
            h = z
        else:
            raise ValueError(f"unknown activation {name!r}")
    return h


def count_linear_regions_1d(params: list[dict], x_min: float, x_max: float) -> int:
    if params[0]["W"].shape[1] != 1:
        raise ValueError("count_linear_regions_1d needs a network with one input")
    for layer in params:
        if layer["activation"] not in ("relu", "identity"):
            raise ValueError("only relu and identity layers are supported")

    # Each stretch is (left, right, slopes, intercepts). On the stretch, every
    # value in the current layer equals slope * x + intercept.
    stretches = [(x_min, x_max, np.array([1.0]), np.array([0.0]))]
    for layer in params:
        W, b = layer["W"], layer["b"]
        is_relu = layer["activation"] == "relu"
        next_stretches = []
        for left, right, slopes, intercepts in stretches:
            pre_slopes = W @ slopes
            pre_intercepts = W @ intercepts + b
            if not is_relu:
                next_stretches.append((left, right, pre_slopes, pre_intercepts))
                continue
            cuts = []
            for s, c in zip(pre_slopes, pre_intercepts):
                if abs(s) > 1e-12:
                    root = -c / s
                    if left + 1e-9 < root < right - 1e-9:
                        cuts.append(root)
            edges = [left]
            for root in sorted(cuts):
                if root - edges[-1] > 1e-9:
                    edges.append(root)
            edges.append(right)
            for a, e in zip(edges[:-1], edges[1:]):
                active = pre_slopes * (0.5 * (a + e)) + pre_intercepts > 0.0
                next_stretches.append(
                    (a, e, np.where(active, pre_slopes, 0.0), np.where(active, pre_intercepts, 0.0))
                )
        stretches = next_stretches

    # Merge neighbours whose final lines agree: such a crossing changed nothing.
    count = 1
    for prev, cur in zip(stretches[:-1], stretches[1:]):
        same = np.allclose(prev[2], cur[2], atol=1e-8) and np.allclose(prev[3], cur[3], atol=1e-8)
        if not same:
            count += 1
    return count


def sawtooth_network(depth: int) -> list[dict]:
    # Layer 1 probes x and x - 1/2. Each later layer folds the previous tent's
    # combination 2 h1 - 4 h2 into its weights and probes it the same way.
    layers = [{"W": np.array([[1.0], [1.0]]), "b": np.array([0.0, -0.5]), "activation": "relu"}]
    for _ in range(depth - 1):
        layers.append(
            {"W": np.array([[2.0, -4.0], [2.0, -4.0]]), "b": np.array([0.0, -0.5]), "activation": "relu"}
        )
    layers.append({"W": np.array([[2.0, -4.0]]), "b": np.array([0.0]), "activation": "identity"})
    return layers


if __name__ == "__main__":
    hinge = [
        {"W": np.array([[1.0]]), "b": np.array([-0.5]), "activation": "relu"},
        {"W": np.array([[1.0]]), "b": np.array([0.0]), "activation": "identity"},
    ]
    print("one hinge, pieces on [0, 1]:", count_linear_regions_1d(hinge, 0.0, 1.0))
    for k in (1, 2, 3, 4):
        net = sawtooth_network(k)
        grid = np.linspace(0.0, 1.0, 11)
        values = [float(mlp_forward(net, np.array([v]))[0]) for v in grid]
        print(f"depth {k}: pieces = {count_linear_regions_1d(net, 0.0, 1.0)}, "
              f"values on a grid of 11 = {np.round(values, 3)}")
