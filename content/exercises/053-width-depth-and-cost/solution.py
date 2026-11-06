"""Reference solution: count the pieces of a folded network, then count its cost."""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)

# Flops charged per output entry for each bend. These are counting conventions.
_BEND_FLOPS = {"relu": 1, "gelu": 9, "identity": 0}


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
    hinge = [
        {"W": np.array([[1.0]]), "b": np.array([-0.5]), "activation": "relu"},
        {"W": np.array([[1.0]]), "b": np.array([0.0]), "activation": "identity"},
    ]
    print("one hinge, pieces on [0, 1]:", count_linear_regions_1d(hinge, 0.0, 1.0))
    for k in (1, 2, 3, 4):
        print(f"depth {k}: pieces = {count_linear_regions_1d(sawtooth_network(k), 0.0, 1.0)}")
    net = _make_net([4, 8, 8, 3], ["relu", "relu", "identity"])
    print("4-8-8-3: parameters", parameter_count(net), "flops", flop_count(net),
          "stored activations", activation_memory(net, 4))
    for k in (1, 2, 5, 10):
        print(f"sawtooth depth {k}: {parameter_count(sawtooth_network(k))} parameters "
              f"(6k + 1 = {6 * k + 1}); one hidden layer would need {3 * 2**k + 1}")
