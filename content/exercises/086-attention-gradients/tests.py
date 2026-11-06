"""Checks for the attention-gradients exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape

QUERY = np.array([1.0, 0.0])
KEYS = np.array([[1.0, 0.0], [0.0, 1.0]])
VALUES = np.array([[10.0, 0.0], [0.0, 10.0]])


def ref_softmax(z):
    e = np.exp(np.asarray(z, dtype=float) - np.max(z))
    return e / e.sum()


def random_case(seed, T=5, d_s=3, d_h=4, h=6, d_v=2):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=d_s),
        rng.normal(size=(T, d_h)),
        rng.normal(size=(T, d_v)),
        0.6 * rng.normal(size=(h, d_s)),
        0.6 * rng.normal(size=(h, d_h)),
        0.6 * rng.normal(size=h),
    )


@check("softmax_backward gives [2.314, -2.314] on the worked example, and it sums to zero")
def test_softmax_backward_worked(sub):
    fn = has(sub, "softmax_backward")
    weights = ref_softmax([0.9640276, 1.5231883])
    dscores = fn(weights, np.array([10.0, 0.0]))
    shape(dscores, (2,), label="dscores")
    close(dscores, np.array([2.3143, -2.3143]), atol=1e-3, label="dscores")
    close(dscores.sum(), 0.0, atol=1e-10, label="sum of dscores")


@check("softmax_backward matches finite differences of softmax(z) @ dweights")
def test_softmax_backward_fd(sub):
    fn = has(sub, "softmax_backward")
    rng = np.random.default_rng(0)
    z = rng.normal(size=6)
    dweights = rng.normal(size=6)
    got = fn(ref_softmax(z), dweights)
    want = finite_difference(lambda arr: float(ref_softmax(arr) @ dweights), z)
    close(got, want, atol=1e-6, rtol=1e-5, label="dscores against finite differences")


@check("softmax_backward returns zeros when every weight gets the same slope")
def test_softmax_backward_constant(sub):
    fn = has(sub, "softmax_backward")
    weights = ref_softmax([0.3, -1.0, 2.0, 0.5])
    got = fn(weights, np.full(4, 7.0))
    close(got, np.zeros(4), atol=1e-12, label="dscores for a constant dweights")


@check("context_backward gives dweights [10, 0] and the weighted rows on the worked example")
def test_context_backward_worked(sub):
    fn = has(sub, "context_backward")
    weights = ref_softmax([0.9640276, 1.5231883])
    dweights, dvalues = fn(np.array([1.0, 0.0]), weights, VALUES)
    shape(dweights, (2,), label="dweights")
    shape(dvalues, (2, 2), label="dvalues")
    close(dweights, np.array([10.0, 0.0]), atol=1e-9, label="dweights")
    close(dvalues, np.array([[0.3637, 0.0], [0.6363, 0.0]]), atol=1e-4, label="dvalues")


@check("context_backward matches finite differences for the weights and the values")
def test_context_backward_fd(sub):
    fn = has(sub, "context_backward")
    rng = np.random.default_rng(1)
    weights = ref_softmax(rng.normal(size=5))
    values = rng.normal(size=(5, 3))
    dcontext = rng.normal(size=3)
    dweights, dvalues = fn(dcontext, weights, values)
    fd_w = finite_difference(lambda w: float((w @ values) @ dcontext), weights)
    fd_v = finite_difference(lambda x: float((weights @ x) @ dcontext), values)
    close(dweights, fd_w, atol=1e-6, rtol=1e-5, label="dweights")
    close(dvalues, fd_v, atol=1e-6, rtol=1e-5, label="dvalues")


@check("attention_backward matches finite differences on all six inputs")
def test_attention_backward_fd(sub):
    forward = has(sub, "additive_attention")
    backward = has(sub, "attention_backward")
    query, keys, values, W1, W2, v = random_case(2)
    dcontext = np.random.default_rng(20).normal(size=values.shape[1])
    grads = backward(dcontext, query, keys, values, W1, W2, v)
    names = ["query", "keys", "values", "W1", "W2", "v"]
    args = {"query": query, "keys": keys, "values": values, "W1": W1, "W2": W2, "v": v}
    if len(grads) != 6:
        fail(f"attention_backward must return 6 gradients, got {len(grads)}")
    for name, grad in zip(names, grads):
        def loss(arr, name=name):
            local = dict(args)
            local[name] = arr
            context, _ = forward(local["query"], local["keys"], local["values"],
                                 local["W1"], local["W2"], local["v"])
            return float(context @ dcontext)

        shape(grad, args[name].shape, label=f"gradient for {name}")
        close(grad, finite_difference(loss, args[name]), atol=1e-6, rtol=1e-4,
              label=f"gradient for {name}")


@check("with one key the weight is always 1, so only the values receive a gradient")
def test_single_key(sub):
    backward = has(sub, "attention_backward")
    query, keys, values, W1, W2, v = random_case(3, T=1)
    dcontext = np.array([0.5, -2.0])
    dquery, dkeys, dvalues, dW1, dW2, dv = backward(dcontext, query, keys, values, W1, W2, v)
    close(dvalues, dcontext[None, :], atol=1e-12, label="dvalues")
    for name, grad in (("dquery", dquery), ("dkeys", dkeys), ("dW1", dW1), ("dW2", dW2), ("dv", dv)):
        close(grad, np.zeros_like(grad), atol=1e-12, label=name)
