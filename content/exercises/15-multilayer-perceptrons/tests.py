"""Checks for the multilayer perceptrons exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def _tent(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.minimum(x, 1.0 - x)


def _sawtooth_closed_form(x: np.ndarray, depth: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    for _ in range(depth):
        y = _tent(y)
    return y


@check("relu matches its finite-difference derivative away from the kink")
def test_relu_grad(sub):
    relu, relu_grad = has(sub, "relu"), has(sub, "relu_grad")
    rng = np.random.default_rng(0)
    x = rng.normal(size=10) * 3.0
    x = np.where(np.abs(x) < 0.05, 0.3, x)  # keep away from the nondifferentiable point

    def total(xv):
        return float(np.sum(relu(xv)))

    fd = finite_difference(total, x)
    close(relu_grad(x), fd, atol=1e-5, label="relu_grad against finite differences")


@check("gelu matches its finite-difference derivative")
def test_gelu_grad(sub):
    gelu, gelu_grad = has(sub, "gelu"), has(sub, "gelu_grad")
    rng = np.random.default_rng(1)
    x = rng.normal(size=10) * 2.0

    def total(xv):
        return float(np.sum(gelu(xv)))

    fd = finite_difference(total, x)
    close(gelu_grad(x), fd, atol=1e-5, label="gelu_grad against finite differences")

    # gelu should be close to zero well below zero and close to identity well above it.
    close(gelu(np.array([-8.0])), np.array([0.0]), atol=1e-4, label="gelu at -8")
    close(gelu(np.array([8.0])), np.array([8.0]), atol=1e-4, label="gelu at 8")


@check("mlp_forward composes an arbitrary list of affine-then-activation layers")
def test_mlp_forward(sub):
    mlp_forward = has(sub, "mlp_forward")
    params = [
        {"W": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), "b": np.array([0.0, 0.0, -1.0]), "activation": "relu"},
        {"W": np.array([[1.0, 1.0, 1.0]]), "b": np.array([0.5]), "activation": "identity"},
    ]
    x = np.array([2.0, -3.0])
    # hidden pre-activation: [2, -3, -2], relu -> [2, 0, 0]; output: 2 + 0.5 = 2.5
    out = mlp_forward(params, x)
    shape(out, (1,), label="mlp_forward output")
    close(out, np.array([2.5]), atol=1e-10, label="mlp_forward on a hand-worked example")


@check("A two-layer network with no nonlinearity is rank-limited to one affine map")
def test_linear_collapse(sub):
    mlp_forward = has(sub, "mlp_forward")
    rng = np.random.default_rng(2)
    in_dim, hidden, out_dim = 6, 3, 5
    W1, b1 = rng.normal(size=(hidden, in_dim)), rng.normal(size=hidden)
    W2, b2 = rng.normal(size=(out_dim, hidden)), rng.normal(size=out_dim)
    params = [
        {"W": W1, "b": b1, "activation": "identity"},
        {"W": W2, "b": b2, "activation": "identity"},
    ]
    composed_W, composed_b = W2 @ W1, W2 @ b1 + b2

    xs = rng.normal(size=(8, in_dim))
    out_net = np.stack([mlp_forward(params, x) for x in xs])
    out_affine = xs @ composed_W.T + composed_b
    close(out_net, out_affine, atol=1e-8, label="two identity layers against the composed affine map")

    rank = np.linalg.matrix_rank(composed_W)
    if rank > hidden:
        fail(
            f"the composed map has rank {rank}, but a bottleneck of width {hidden} "
            f"can carry rank at most {hidden}"
        )


@check("sawtooth_network(k) produces exactly 2**k linear pieces on [0, 1]")
def test_sawtooth_pieces(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    count_linear_regions_1d = has(sub, "count_linear_regions_1d")
    for depth in (1, 2, 3, 4):
        net = sawtooth_network(depth)
        regions = count_linear_regions_1d(net, 0.0, 1.0)
        if regions != 2**depth:
            fail(f"depth {depth}: got {regions} linear pieces, expected {2**depth}")


@check("sawtooth_network(k) matches the closed-form sawtooth to machine precision")
def test_sawtooth_closed_form(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    mlp_forward = has(sub, "mlp_forward")
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.0, 1.0, size=40)
    for depth in (1, 2, 3, 4, 5):
        net = sawtooth_network(depth)
        got = np.array([mlp_forward(net, np.array([x]))[0] for x in xs])
        want = _sawtooth_closed_form(xs, depth)
        close(got, want, atol=1e-10, label=f"depth {depth}")


@check("parameter_count matches a hand-computed value for a stated architecture")
def test_parameter_count(sub):
    parameter_count = has(sub, "parameter_count")
    rng = np.random.default_rng(4)
    dims, activations = [4, 8, 8, 3], ["relu", "relu", "identity"]
    params = [
        {"W": rng.normal(size=(dout, din)), "b": rng.normal(size=dout), "activation": act}
        for din, dout, act in zip(dims[:-1], dims[1:], activations)
    ]
    # (4*8 + 8) + (8*8 + 8) + (8*3 + 3) = 40 + 72 + 27 = 139
    close(parameter_count(params), 139, label="parameter_count on a 4-8-8-3 network")


@check("flop_count matches a hand-computed value for a stated architecture")
def test_flop_count(sub):
    flop_count = has(sub, "flop_count")
    rng = np.random.default_rng(4)
    dims, activations = [4, 8, 8, 3], ["relu", "relu", "identity"]
    params = [
        {"W": rng.normal(size=(dout, din)), "b": rng.normal(size=dout), "activation": act}
        for din, dout, act in zip(dims[:-1], dims[1:], activations)
    ]
    # layer 1: 2*4*8 + 8 + 8*1 = 64 + 8 + 8 = 80
    # layer 2: 2*8*8 + 8 + 8*1 = 128 + 8 + 8 = 144
    # layer 3: 2*8*3 + 3 + 0   = 48 + 3 = 51
    # total: 80 + 144 + 51 = 275
    close(flop_count(params), 275, label="flop_count on a 4-8-8-3 network")
