"""Checks for the load-balancing exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@check("expert_load counts slots and divides by n * k, so it sums to one")
def test_load(sub):
    f = has(sub, "expert_load")
    idx = np.array([[0, 1], [0, 2], [0, 1]])  # n = 3, k = 2, six slots
    got = f(idx, 4)
    shape(got, (4,), label="f")
    close(got, np.array([3 / 6, 2 / 6, 1 / 6, 0.0]), atol=1e-12, label="f")
    close(got.sum(), 1.0, atol=1e-12, label="sum of f")


@check("mean_gate_probability averages over tokens and sums to one")
def test_mean(sub):
    f = has(sub, "mean_gate_probability")
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    got = f(probs)
    close(got, np.array([0.75, 0.25]), atol=1e-12, label="P")
    rng = np.random.default_rng(0)
    big = _softmax(rng.normal(size=(9, 5)))
    shape(f(big), (5,), label="P")
    close(f(big).sum(), 1.0, atol=1e-12, label="sum of P")


@check("the worked example: both tokens to expert 0 gives loss 1.5")
def test_hand(sub):
    f = has(sub, "load_balancing_loss")
    idx = np.array([[0], [0]])
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    close(f(idx, probs, 2), 1.5, atol=1e-12, label="loss")


@check("uniform routing scores exactly 1")
def test_uniform(sub):
    f = has(sub, "load_balancing_loss")
    E = 4
    idx = np.array([[0], [1], [2], [3]])
    close(f(idx, np.full((4, E), 1 / E), E), 1.0, atol=1e-12, label="uniform, k = 1")
    idx2 = np.array([[0, 1], [2, 3], [0, 1], [2, 3]])
    close(f(idx2, np.full((4, E), 1 / E), E), 1.0, atol=1e-12, label="uniform, k = 2")


@check("total collapse scores E for k = 1 and E / k for larger k")
def test_collapse(sub):
    f = has(sub, "load_balancing_loss")
    E, n = 4, 6
    one_hot = np.zeros((n, E))
    one_hot[:, 0] = 1.0
    close(f(np.zeros((n, 1), dtype=int), one_hot, E), 4.0, atol=1e-12, label="collapse, k = 1")
    two_hot = np.zeros((n, E))
    two_hot[:, :2] = 0.5
    idx = np.tile(np.array([0, 1]), (n, 1))
    close(f(idx, two_hot, E), 2.0, atol=1e-12, label="collapse onto two experts, k = 2")


@check("random routings never exceed E / k, and a busier favourite scores higher")
def test_bounds(sub):
    f = has(sub, "load_balancing_loss")
    rng = np.random.default_rng(3)
    E, n, k = 6, 40, 2
    for _ in range(5):
        probs = _softmax(rng.normal(size=(n, E)) * 2)
        idx = np.argsort(-probs, axis=1)[:, :k]
        loss = f(idx, probs, E)
        if loss > E / k + 1e-9 or loss < 0:
            fail(f"loss {loss:.4f} is outside [0, {E / k}] for k = {k}, E = {E}")
    # Same probabilities, but the choices pile onto expert 0: the loss goes up.
    probs = np.tile(np.array([0.4, 0.3, 0.2, 0.1]), (8, 1))
    spread = np.array([[0], [1], [2], [3], [0], [1], [2], [3]])
    piled = np.zeros((8, 1), dtype=int)
    if not f(piled, probs, 4) > f(spread, probs, 4):
        fail("sending every token to the most probable expert should score higher than spreading them")
