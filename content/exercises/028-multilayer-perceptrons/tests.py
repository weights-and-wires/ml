"""Checks for the multilayer perceptron exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


def _gelu_ref(x):
    u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


def _net(dims, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]


def _one_hidden(kinks, out_weights, in_weight=1.0):
    n = len(kinks)
    return [
        {"W": np.full((n, 1), in_weight), "b": -np.asarray(kinks, dtype=float) * in_weight, "activation": "relu"},
        {"W": np.asarray(out_weights, dtype=float).reshape(1, n), "b": np.array([0.0]), "activation": "identity"},
    ]


def _tent(x):
    return 2.0 * np.minimum(x, 1.0 - x)


@check("relu and relu_grad: values, shapes, and finite differences away from the corner")
def test_relu(sub):
    relu, relu_grad = has(sub, "relu"), has(sub, "relu_grad")
    out = relu(np.array([2.0, -3.0, -2.0]))
    close(out, np.array([2.0, 0.0, 0.0]), atol=1e-12, label="relu on [2, -3, -2]")
    m = np.array([[1.5, -0.5], [0.0, -7.0]])
    shape(relu(m), (2, 2), label="relu on a matrix")
    close(relu(m), np.array([[1.5, 0.0], [0.0, 0.0]]), atol=1e-12, label="relu on a matrix")
    g = relu_grad(np.array([2.0, -3.0, -2.0]))
    if g.dtype.kind != "f":
        fail(f"relu_grad should return floats, got dtype {g.dtype}")
    close(g, np.array([1.0, 0.0, 0.0]), atol=1e-12, label="relu_grad on [2, -3, -2]")
    rng = np.random.default_rng(0)
    x = rng.normal(size=12) * 3.0
    x = np.where(np.abs(x) < 0.05, 0.3, x)
    fd = finite_difference(lambda v: float(np.sum(relu(v))), x)
    close(relu_grad(x), fd, atol=1e-5, label="relu_grad against finite differences")


@check("compose_affine: the hand example, two random layers, and the rank limit")
def test_compose(sub):
    compose_affine = has(sub, "compose_affine")
    W, b = compose_affine(
        np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0]),
        np.array([[3.0, 1.0]]), np.array([4.0]),
    )
    shape(W, (1, 2), label="merged W")
    close(W, np.array([[3.0, 2.0]]), atol=1e-12, label="merged W")
    close(b, np.array([7.0]), atol=1e-12, label="merged b")

    rng = np.random.default_rng(1)
    W1, b1 = rng.normal(size=(4, 6)), rng.normal(size=4)
    W2, b2 = rng.normal(size=(3, 4)), rng.normal(size=3)
    W, b = compose_affine(W1, b1, W2, b2)
    shape(W, (3, 6), label="merged W")
    shape(b, (3,), label="merged b")
    X = rng.normal(size=(10, 6))
    close(X @ W.T + b, (X @ W1.T + b1) @ W2.T + b2, atol=1e-10, label="merged layer against two layers")

    rng = np.random.default_rng(2)
    W1, b1 = rng.normal(size=(2, 6)), rng.normal(size=2)
    W2, b2 = rng.normal(size=(5, 2)), rng.normal(size=5)
    W, _ = compose_affine(W1, b1, W2, b2)
    rank = np.linalg.matrix_rank(W)
    if rank > 2:
        fail(f"merged weights have rank {rank}, but a middle layer of 2 units allows at most 2")
    if rank < 2:
        fail(f"merged weights have rank {rank}; random layers with a middle width of 2 should reach 2")


@check("Two ReLU units solve XOR, and a ReLU between two layers changes the answer to 13")
def test_xor_and_bend(sub):
    relu = has(sub, "relu")
    compose_affine = has(sub, "compose_affine")
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    on = np.array([False, False, True, True])
    W_hidden = np.array([[1.0, -1.0], [-1.0, 1.0]])
    score = relu(points @ W_hidden.T).sum(axis=1) - 0.5
    if not np.all((score > 0) == on):
        fail(f"h1 + h2 - 0.5 should be positive only for the lamp-on points, got {score}")

    W1, b1 = np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0])
    W2, b2 = np.array([[3.0, 1.0]]), np.array([4.0])
    x = np.array([2.0, -3.0])
    W, b = compose_affine(W1, b1, W2, b2)
    merged = W @ x + b
    with_bend = W2 @ relu(W1 @ x + b1) + b2
    close(with_bend, np.array([13.0]), atol=1e-12, label="two layers with a ReLU between them")
    if np.allclose(merged, with_bend):
        fail("the merged layer should disagree with the network that has a ReLU in the middle")


@check("gelu matches the hand values at -1, 0, and 1, and follows ReLU far from zero")
def test_gelu_values(sub):
    gelu = has(sub, "gelu")
    out = gelu(np.array([-1.0, 0.0, 1.0]))
    shape(out, (3,), label="gelu output")
    close(out, np.array([-0.158808, 0.0, 0.841192]), atol=1e-6, label="gelu at -1, 0, 1")
    close(gelu(np.array([-8.0])), np.array([0.0]), atol=1e-4, label="gelu at -8")
    close(gelu(np.array([8.0])), np.array([8.0]), atol=1e-4, label="gelu at 8")
    m = np.array([[0.5, -2.0], [3.0, -0.25]])
    shape(gelu(m), (2, 2), label="gelu on a matrix")


@check("gelu_grad matches finite differences and the hand values at 0 and 1")
def test_gelu_grad(sub):
    gelu, gelu_grad = has(sub, "gelu"), has(sub, "gelu_grad")
    rng = np.random.default_rng(1)
    x = rng.normal(size=12) * 2.0
    fd = finite_difference(lambda v: float(np.sum(gelu(v))), x)
    close(gelu_grad(x), fd, atol=1e-5, label="gelu_grad against finite differences")
    close(gelu_grad(np.array([0.0, 1.0])), np.array([0.5, 1.082964]), atol=1e-6, label="gelu_grad at 0 and 1")


@check("mlp_forward: the hand example, a GELU layer, mixed bends, and an unknown name")
def test_forward(sub):
    mlp_forward = has(sub, "mlp_forward")
    params = [
        {"W": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), "b": np.array([0.0, 0.0, -1.0]), "activation": "relu"},
        {"W": np.array([[1.0, 1.0, 1.0]]), "b": np.array([0.5]), "activation": "identity"},
    ]
    out = mlp_forward(params, np.array([2.0, -3.0]))
    shape(out, (1,), label="mlp_forward output")
    close(out, np.array([2.5]), atol=1e-10, label="mlp_forward on the hand example")

    rng = np.random.default_rng(2)
    W1, b1 = rng.normal(size=(5, 3)), rng.normal(size=5)
    W2, b2 = rng.normal(size=(2, 5)), rng.normal(size=2)
    x = rng.normal(size=3)
    gelu_net = [
        {"W": W1, "b": b1, "activation": "gelu"},
        {"W": W2, "b": b2, "activation": "identity"},
    ]
    want = W2 @ _gelu_ref(W1 @ x + b1) + b2
    out = mlp_forward(gelu_net, x)
    shape(out, (2,), label="mlp_forward output")
    close(out, want, atol=1e-10, label="mlp_forward with a GELU layer")

    rng = np.random.default_rng(3)
    dims = [4, 6, 5, 3]
    acts = ["relu", "gelu", "identity"]
    mixed = [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]
    x = rng.normal(size=4)
    h = np.maximum(mixed[0]["W"] @ x + mixed[0]["b"], 0.0)
    h = _gelu_ref(mixed[1]["W"] @ h + mixed[1]["b"])
    want = mixed[2]["W"] @ h + mixed[2]["b"]
    close(mlp_forward(mixed, x), want, atol=1e-10, label="three different bends in a row")

    bad = [{"W": np.eye(2), "b": np.zeros(2), "activation": "swish"}]
    try:
        mlp_forward(bad, np.array([1.0, 2.0]))
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001
        fail(f"an unknown activation should raise ValueError, got {type(exc).__name__}")
    fail("an unknown activation name should raise ValueError, but nothing was raised")


@check("count_linear_regions_1d: hinges, merged crossings, and five kinks")
def test_pieces(sub):
    count = has(sub, "count_linear_regions_1d")
    equal(int(count(_one_hidden([0.5], [1.0]), 0.0, 1.0)), 2, label="pieces of relu(x - 0.5)")
    linear = [
        {"W": np.array([[2.0], [-1.0]]), "b": np.array([1.0, 0.5]), "activation": "identity"},
        {"W": np.array([[3.0, 1.0]]), "b": np.array([4.0]), "activation": "identity"},
    ]
    equal(int(count(linear, 0.0, 1.0)), 1, label="identity-only network")
    equal(int(count(_one_hidden([5.0], [1.0]), 0.0, 1.0)), 1, label="relu(x - 5) on [0, 1], always cut to 0")
    equal(int(count(_one_hidden([-5.0], [1.0]), 0.0, 1.0)), 1, label="relu(x + 5) on [0, 1], always passes")
    equal(int(count(_one_hidden([0.3, 0.7], [1.0, 0.0]), 0.0, 1.0)), 2, label="the unit with output weight 0 adds nothing")
    equal(int(count(_one_hidden([0.5], [0.0]), 0.0, 1.0)), 1, label="a hinge whose output weight is 0")
    net = _one_hidden([0.1, 0.3, 0.5, 0.7, 0.9], [1.0, -2.0, 1.5, 1.0, -1.0])
    equal(int(count(net, 0.0, 1.0)), 6, label="5 hinges, 6 pieces")
    equal(int(count(net, 0.0, 0.4)), 3, label="the interval [0, 0.4] holds two kinks")


@check("sawtooth_network(k) matches the tent map applied k times and has 2**k pieces")
def test_sawtooth(sub):
    sawtooth_network = has(sub, "sawtooth_network")
    mlp_forward = has(sub, "mlp_forward")
    count = has(sub, "count_linear_regions_1d")
    for depth in (1, 2, 4):
        net = sawtooth_network(depth)
        equal(len(net), depth + 1, label=f"layers at depth {depth}")
        for layer in net[:-1]:
            equal(layer["W"].shape[0], 2, label="hidden width")
            equal(layer["activation"], "relu", label="hidden activation")
        equal(net[-1]["W"].shape, (1, 2), label="output W shape")
        equal(net[-1]["activation"], "identity", label="output activation")
        equal(net[0]["W"].shape, (2, 1), label="first W shape")
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.0, 1.0, size=30)
    for depth in (1, 2, 3, 4, 5):
        net = sawtooth_network(depth)
        got = np.array([mlp_forward(net, np.array([x]))[0] for x in xs])
        want = xs.copy()
        for _ in range(depth):
            want = _tent(want)
        close(got, want, atol=1e-10, label=f"depth {depth}")
        pieces = int(count(net, 0.0, 1.0))
        if pieces != 2**depth:
            fail(f"depth {depth}: got {pieces} pieces, expected {2**depth}")


@check("parameter_count and flop_count: 139 parameters and 275 flops for 4-8-8-3, 6k + 1 for the sawtooth")
def test_params_and_flops(sub):
    parameter_count = has(sub, "parameter_count")
    flop_count = has(sub, "flop_count")
    sawtooth_network = has(sub, "sawtooth_network")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    # (4*8 + 8) + (8*8 + 8) + (8*3 + 3) = 40 + 72 + 27
    equal(int(parameter_count(net)), 139, label="parameters of 4-8-8-3")
    equal(int(parameter_count(_net([4, 16, 16, 3], ["relu", "relu", "identity"]))), 403, label="parameters of 4-16-16-3")
    for k in range(1, 8):
        equal(int(parameter_count(sawtooth_network(k))), 6 * k + 1, label=f"parameters at depth {k}")
    for k in (1, 2, 4, 6):
        n = 2**k
        got = int(parameter_count(_net([1, n, 1], ["relu", "identity"])))
        if got != 3 * n + 1:
            fail(f"k = {k}: a 1-{n}-1 network has {got} parameters, expected {3 * n + 1}")
    # 80 + 144 + 51
    equal(int(flop_count(net)), 275, label="flops of 4-8-8-3")
    # layer 1: 2*4*8 + 8 + 9*8 = 144; layer 2: 2*8*3 + 3 + 0 = 51
    equal(int(flop_count(_net([4, 8, 3], ["gelu", "identity"]))), 195, label="flops of a 4-8-3 network with a GELU layer")
    # a swapped shape would give 2*3*10 + 3 + 3 = 66
    equal(int(flop_count(_net([3, 10], ["relu"]))), 80, label="flops of one 3-to-10 ReLU layer")
    a = _net([5, 7], ["relu"], seed=1)
    b = _net([7, 4, 2], ["gelu", "identity"], seed=2)
    equal(int(flop_count(a + b)), int(flop_count(a)) + int(flop_count(b)), label="flops of the stack")


@check("activation_memory counts the input and every layer's output")
def test_activation_memory(sub):
    activation_memory = has(sub, "activation_memory")
    sawtooth_network = has(sub, "sawtooth_network")
    net = _net([4, 8, 8, 3], ["relu", "relu", "identity"])
    equal(int(activation_memory(net, 4)), 23, label="stored activations of 4-8-8-3")
    for k in (1, 3, 6):
        # input 1, then 2 per hidden layer, then 1 output
        equal(int(activation_memory(sawtooth_network(k), 1)), 2 * k + 2, label=f"sawtooth depth {k}")


def _run(params, x):
    """Evaluate a list of layers at scalar inputs with plain numpy, so these
    checks do not depend on the reader's own mlp_forward."""
    h = np.asarray(x, dtype=float).reshape(-1, 1)
    for layer in params:
        z = h @ np.asarray(layer["W"]).T + np.asarray(layer["b"])
        h = np.maximum(z, 0.0) if layer["activation"] == "relu" else z
    return h[:, 0]


@check("hinge_interpolant: the hand example (output weights 2, -3, 1) and exact heights at random knots")
def test_hinge_hand_and_knots(sub):
    # Wrong implementation caught here: using each segment's slope as the output
    # weight, [2, -1, 0], instead of the change in slope, [2, -3, 1]. It gives
    # 5 instead of 2 at x = 3, and the knots after the second one drift away.
    hinge_interpolant = has(sub, "hinge_interpolant")
    xs = np.array([0.0, 1.0, 2.0, 4.0])
    ys = np.array([1.0, 3.0, 2.0, 2.0])
    net = hinge_interpolant(xs, ys)
    equal(len(net), 2, label="layers of the hinge network")
    shape(net[0]["W"], (3, 1), label="hidden W")
    shape(net[0]["b"], (3,), label="hidden b")
    shape(net[1]["W"], (1, 3), label="output W")
    equal(net[0]["activation"], "relu", label="hidden activation")
    equal(net[1]["activation"], "identity", label="output activation")
    close(net[0]["b"], np.array([0.0, -1.0, -2.0]), atol=1e-12, label="hidden biases, minus each left knot")
    close(net[1]["W"].reshape(-1), np.array([2.0, -3.0, 1.0]), atol=1e-12, label="output weights, the change in slope")
    close(net[1]["b"].reshape(-1), np.array([1.0]), atol=1e-12, label="output bias, the first height")
    close(_run(net, [3.0]), np.array([2.0]), atol=1e-9, label="network at x = 3, on the flat segment")

    rng = np.random.default_rng(7)
    xs = np.sort(rng.uniform(-3.0, 5.0, size=9))
    ys = rng.normal(size=9)
    net = hinge_interpolant(xs, ys)
    close(_run(net, xs), ys, atol=1e-9, label="network at every random knot")


@check("hinge_interpolant is straight between knots, with offsets, flat data, and bad knots")
def test_hinge_between_knots(sub):
    # Wrong implementations caught here: a hinge placed at +x_j instead of -x_j
    # (knots away from 0 break), and a missing output bias (shifts every height
    # by ys[0], which is 4 here).
    hinge_interpolant = has(sub, "hinge_interpolant")
    xs = np.array([-2.0, -1.5, 0.5, 1.0, 3.5])
    ys = np.array([4.0, 1.0, 2.5, -1.0, 0.0])
    net = hinge_interpolant(xs, ys)
    mids = 0.5 * (xs[:-1] + xs[1:])
    quarters = xs[:-1] + 0.25 * np.diff(xs)
    probe = np.concatenate([mids, quarters, xs])
    close(_run(net, probe), np.interp(probe, xs, ys), atol=1e-9, label="network against np.interp between knots")

    flat = hinge_interpolant(np.array([0.0, 1.0, 2.0]), np.array([2.0, 2.0, 2.0]))
    close(flat[1]["W"].reshape(-1), np.zeros(2), atol=1e-12, label="output weights of a constant function")
    close(_run(flat, [0.3, 1.7]), np.array([2.0, 2.0]), atol=1e-12, label="constant function")

    two = hinge_interpolant(np.array([1.0, 3.0]), np.array([0.0, 4.0]))
    shape(two[0]["W"], (1, 1), label="hidden W for two knots")
    close(_run(two, [1.0, 2.0, 3.0]), np.array([0.0, 2.0, 4.0]), atol=1e-12, label="one segment, slope 2")

    for bad_xs, bad_ys in (
        (np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0])),
        (np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0])),
        (np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])),
    ):
        try:
            hinge_interpolant(bad_xs, bad_ys)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            fail(f"knots {bad_xs.tolist()} should raise ValueError, got {type(exc).__name__}")
        fail(f"knots {bad_xs.tolist()} with {bad_ys.size} heights should raise ValueError, but nothing was raised")


@check("hinge_interpolant on sin over [0, pi]: the maximum error falls about 4x each time the knots double")
def test_hinge_rate(sub):
    # The slopes-as-output-weights version misses the knots, so it fails the
    # knot check below before the rate is measured.
    hinge_interpolant = has(sub, "hinge_interpolant")
    grid = np.linspace(0.0, np.pi, 20001)
    errors = []
    for segments in (8, 16, 32, 64):
        xs = np.linspace(0.0, np.pi, segments + 1)
        net = hinge_interpolant(xs, np.sin(xs))
        close(_run(net, xs), np.sin(xs), atol=1e-9, label=f"{segments} segments, at the knots")
        err = float(np.max(np.abs(_run(net, grid) - np.sin(grid))))
        bound = (np.pi / segments) ** 2 / 8.0  # h**2 / 8 times max |sin''| = 1
        if err > bound * 1.001:
            fail(f"{segments} segments: max error {err:.6g} exceeds the h**2/8 bound {bound:.6g}")
        errors.append(err)
    for (a, b), segments in zip(zip(errors[:-1], errors[1:]), (8, 16, 32)):
        ratio = a / b
        if not 3.8 <= ratio <= 4.2:
            fail(f"doubling {segments} -> {2 * segments} segments: error ratio {ratio:.3f}, expected about 4 (in [3.8, 4.2])")
