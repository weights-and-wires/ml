"""Reference solution: a multilayer perceptron, its pieces, and its cost."""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_GELU_CUBIC = 0.044715

# Flops charged per output entry for each bend. These are counting conventions.
_BEND_FLOPS = {"relu": 1, "gelu": 9, "identity": 0}


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def compose_affine(W1, b1, W2, b2):
    # W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
    return W2 @ W1, W2 @ b1 + b2


def gelu(x: np.ndarray) -> np.ndarray:
    u = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    u = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    t = np.tanh(u)
    du = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_CUBIC * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du


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


def parameter_count(params: list[dict]) -> int:
    return int(sum(layer["W"].size + layer["b"].size for layer in params))


def flop_count(params: list[dict]) -> int:
    total = 0
    for layer in params:
        out_dim, in_dim = layer["W"].shape
        total += 2 * in_dim * out_dim  # one multiply and one add per weight
        total += out_dim  # bias additions
        total += _BEND_FLOPS[layer["activation"]] * out_dim
    return int(total)


def activation_memory(params: list[dict], in_dim: int) -> int:
    return int(in_dim + sum(layer["W"].shape[0] for layer in params))


def _make_net(dims, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]


if __name__ == "__main__":
    # XOR: lamp on when exactly one switch is up.
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    W_hidden = np.array([[1.0, -1.0], [-1.0, 1.0]])  # rows: x1 - x2 and x2 - x1
    hidden = relu(points @ W_hidden.T)
    print("hidden features:\n", hidden)
    print("h1 + h2 - 0.5:", hidden.sum(axis=1) - 0.5)

    W, b = compose_affine(
        np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0]),
        np.array([[3.0, 1.0]]), np.array([4.0]),
    )
    print("merged W:", W, "merged b:", b)
    print("gelu(1), gelu(-1):", gelu(np.array([1.0, -1.0])))
    print("gelu_grad(1):", gelu_grad(np.array([1.0])))

    params = [
        {"W": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
         "b": np.array([0.0, 0.0, -1.0]), "activation": "relu"},
        {"W": np.array([[1.0, 1.0, 1.0]]), "b": np.array([0.5]), "activation": "identity"},
    ]
    print("network output:", mlp_forward(params, np.array([2.0, -3.0])))

    for k in (1, 2, 3, 4):
        print(f"depth {k}: pieces = {count_linear_regions_1d(sawtooth_network(k), 0.0, 1.0)}")
    net = _make_net([4, 8, 8, 3], ["relu", "relu", "identity"])
    print("4-8-8-3: parameters", parameter_count(net), "flops", flop_count(net),
          "stored activations", activation_memory(net, 4))
