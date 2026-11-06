"""Checks for the forward-pass and GELU exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def _gelu_ref(x):
    u = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(u))


@check("gelu matches the hand values at -1, 0, and 1")
def test_gelu_values(sub):
    gelu = has(sub, "gelu")
    out = gelu(np.array([-1.0, 0.0, 1.0]))
    shape(out, (3,), label="gelu output")
    close(out, np.array([-0.158808, 0.0, 0.841192]), atol=1e-6, label="gelu at -1, 0, 1")


@check("gelu follows ReLU far from zero")
def test_gelu_limits(sub):
    gelu = has(sub, "gelu")
    close(gelu(np.array([-8.0])), np.array([0.0]), atol=1e-4, label="gelu at -8")
    close(gelu(np.array([8.0])), np.array([8.0]), atol=1e-4, label="gelu at 8")
    m = np.array([[0.5, -2.0], [3.0, -0.25]])
    shape(gelu(m), (2, 2), label="gelu on a matrix")


@check("gelu_grad matches finite differences")
def test_gelu_grad_fd(sub):
    gelu, gelu_grad = has(sub, "gelu"), has(sub, "gelu_grad")
    rng = np.random.default_rng(1)
    x = rng.normal(size=12) * 2.0
    fd = finite_difference(lambda v: float(np.sum(gelu(v))), x)
    close(gelu_grad(x), fd, atol=1e-5, label="gelu_grad against finite differences")


@check("gelu_grad hits the hand values at 0 and 1")
def test_gelu_grad_values(sub):
    gelu_grad = has(sub, "gelu_grad")
    close(gelu_grad(np.array([0.0, 1.0])), np.array([0.5, 1.082964]), atol=1e-6, label="gelu_grad at 0 and 1")


@check("mlp_forward reproduces the hand-worked example, 2.5")
def test_forward_hand(sub):
    mlp_forward = has(sub, "mlp_forward")
    params = [
        {"W": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), "b": np.array([0.0, 0.0, -1.0]), "activation": "relu"},
        {"W": np.array([[1.0, 1.0, 1.0]]), "b": np.array([0.5]), "activation": "identity"},
    ]
    out = mlp_forward(params, np.array([2.0, -3.0]))
    shape(out, (1,), label="mlp_forward output")
    close(out, np.array([2.5]), atol=1e-10, label="mlp_forward on the hand example")


@check("mlp_forward handles rectangular weights and a GELU layer")
def test_forward_gelu(sub):
    mlp_forward = has(sub, "mlp_forward")
    rng = np.random.default_rng(2)
    W1, b1 = rng.normal(size=(5, 3)), rng.normal(size=5)
    W2, b2 = rng.normal(size=(2, 5)), rng.normal(size=2)
    x = rng.normal(size=3)
    params = [
        {"W": W1, "b": b1, "activation": "gelu"},
        {"W": W2, "b": b2, "activation": "identity"},
    ]
    want = _gelu_ref(W1 @ x + b1)
    want = W2 @ want + b2
    out = mlp_forward(params, x)
    shape(out, (2,), label="mlp_forward output")
    close(out, want, atol=1e-10, label="mlp_forward with a GELU layer")


@check("mlp_forward mixes relu, gelu, and identity, and rejects unknown names")
def test_forward_mixed(sub):
    mlp_forward = has(sub, "mlp_forward")
    rng = np.random.default_rng(3)
    dims = [4, 6, 5, 3]
    acts = ["relu", "gelu", "identity"]
    params = [
        {"W": rng.normal(size=(o, i)), "b": rng.normal(size=o), "activation": a}
        for i, o, a in zip(dims[:-1], dims[1:], acts)
    ]
    x = rng.normal(size=4)
    h = np.maximum(params[0]["W"] @ x + params[0]["b"], 0.0)
    h = _gelu_ref(params[1]["W"] @ h + params[1]["b"])
    want = params[2]["W"] @ h + params[2]["b"]
    close(mlp_forward(params, x), want, atol=1e-10, label="three different bends in a row")
    bad = [{"W": np.eye(2), "b": np.zeros(2), "activation": "swish"}]
    try:
        mlp_forward(bad, np.array([1.0, 2.0]))
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001
        fail(f"an unknown activation should raise ValueError, got {type(exc).__name__}")
    fail("an unknown activation name should raise ValueError, but nothing was raised")
