"""Checks for the router gradient exercise."""

import numpy as np

from mlcheck import check, close, finite_difference, fail, has, shape


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _surrogate(x, W, k, outs, go):
    """The test's own surrogate loss, independent of the reader's code."""
    probs = _softmax(x @ W)
    total = 0.0
    for t in range(x.shape[0]):
        idx = np.argsort(-(x[t] @ W))[:k]
        g = probs[t, idx] / probs[t, idx].sum()
        total += float(g @ (outs[t, idx] @ go[t]))
    return total


@check("gate_logit_grad matches the worked example: -0.3932 and +0.3932")
def test_gate_hand(sub):
    f = has(sub, "gate_logit_grad")
    g = _softmax(np.array([1.0, 0.0]))
    out = f(g, np.array([1.0, 3.0]))
    close(out, np.array([-0.39322387, 0.39322387]), atol=1e-6, label="gate_logit_grad")


@check("gate_logit_grad sums to zero, and is zero when every c is equal or k = 1")
def test_gate_structure(sub):
    f = has(sub, "gate_logit_grad")
    rng = np.random.default_rng(0)
    g = _softmax(rng.normal(size=4))
    c = rng.normal(size=4)
    close(np.sum(f(g, c)), 0.0, atol=1e-9, label="sum of the gradients")
    close(f(g, np.full(4, 2.5)), np.zeros(4), atol=1e-9, label="gradient when all c are equal")
    close(f(np.array([1.0]), np.array([3.0])), np.zeros(1), atol=1e-9, label="gradient when k = 1")


@check("gate_logit_grad matches finite differences through the softmax")
def test_gate_fd(sub):
    f = has(sub, "gate_logit_grad")
    rng = np.random.default_rng(1)
    z = rng.normal(size=3)
    c = rng.normal(size=3)
    fd = finite_difference(lambda zz: float(_softmax(zz) @ c), z)
    close(f(_softmax(z), c), fd, rtol=1e-5, atol=1e-6, label="gate_logit_grad vs finite differences")


@check("surrogate_loss reproduces the hand-computed mix 5.614 and ignores unchosen experts")
def test_surrogate_hand(sub):
    f = has(sub, "surrogate_loss")
    x = np.array([[1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    go = np.array([[1.0]])
    outs = np.array([[[4.0], [10.0], [100.0]]])
    got = f(x, W, 2, outs, go)
    close(got, 5.61364853, atol=1e-6, label="surrogate_loss")
    outs2 = np.array([[[4.0], [10.0], [-5e8]]])
    close(f(x, W, 2, outs2, go), got, atol=1e-6, label="surrogate_loss after changing an unchosen expert")


@check("router_backward returns shape (d, E) and zero when k = 1")
def test_backward_shape_k1(sub):
    f = has(sub, "router_backward")
    rng = np.random.default_rng(2)
    n, d, E, d_out = 4, 3, 5, 2
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    outs = rng.normal(size=(n, E, d_out))
    go = rng.normal(size=(n, d_out))
    shape(f(x, W, 2, outs, go), (d, E), label="router_backward")
    close(f(x, W, 1, outs, go), np.zeros((d, E)), atol=1e-9, label="router_backward when k = 1")


@check("router_backward matches finite differences of the surrogate loss")
def test_backward_fd(sub):
    f = has(sub, "router_backward")
    rng = np.random.default_rng(7)
    n, d, E, k, d_out = 6, 4, 4, 2, 3
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E)) * 1.5
    outs = rng.normal(size=(n, E, d_out))
    go = rng.normal(size=(n, d_out))
    fd = finite_difference(lambda w: _surrogate(x, w.reshape(d, E), k, outs, go), W.flatten()).reshape(d, E)
    close(f(x, W, k, outs, go), fd, rtol=1e-3, atol=1e-4, label="router_backward vs finite differences")


@check("surrogate_loss and router_backward agree with each other")
def test_agree(sub):
    loss = has(sub, "surrogate_loss")
    back = has(sub, "router_backward")
    rng = np.random.default_rng(11)
    n, d, E, k, d_out = 5, 3, 4, 3, 2
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    outs = rng.normal(size=(n, E, d_out))
    go = rng.normal(size=(n, d_out))
    fd = finite_difference(lambda w: loss(x, w.reshape(d, E), k, outs, go), W.flatten()).reshape(d, E)
    close(back(x, W, k, outs, go), fd, rtol=1e-3, atol=1e-4, label="router_backward vs your surrogate_loss")
