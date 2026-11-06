"""Checks for the balancing exercise: the auxiliary loss, capacity, and dropping."""

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


@check("the worked example gives loss 1.5, and uniform routing scores exactly 1")
def test_hand(sub):
    f = has(sub, "load_balancing_loss")
    idx = np.array([[0], [0]])
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    close(f(idx, probs, 2), 1.5, atol=1e-12, label="loss")
    E = 4
    idx_even = np.array([[0], [1], [2], [3]])
    close(f(idx_even, np.full((4, E), 1 / E), E), 1.0, atol=1e-12, label="uniform, k = 1")
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


def _setup(seed=0, n=12, d=4, E=3, d_out=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    W[:, 0] += 1.5  # expert 0 is popular, so it overflows
    experts = [rng.normal(size=(d, d_out)) for _ in range(E)]
    return x, W, experts


@check("expert_capacity gives 4 and 2 on the worked example, n*k for None, rounds up, and never returns less than 1")
def test_capacity(sub):
    f = has(sub, "expert_capacity")
    equal(f(6, 2, 3, 1.0), 4, label="capacity at gamma = 1")
    equal(f(6, 2, 3, 0.5), 2, label="capacity at gamma = 0.5")
    equal(f(6, 2, 3, None), 12, label="capacity when unlimited")
    equal(f(5, 1, 4, 1.0), 2, label="ceil(1.25)")
    equal(f(2, 1, 8, 0.5), 1, label="a tiny factor still leaves one seat")
    equal(f(8, 1, 4, 2.0), 4, label="eight tokens, four experts, gamma 2")

@check("dropped_fraction divides by all n * k assignments")
def test_fraction(sub):
    f = has(sub, "dropped_fraction")
    close(f(6, 12, 2), 0.25, atol=1e-12, label="6 of 24")
    close(f(0, 10, 3), 0.0, atol=1e-12, label="nothing dropped")


@check("moe_forward drops exactly the overflow when expert 0 is oversubscribed")
def test_counts(sub):
    fwd = has(sub, "moe_forward")
    router = has(sub, "router")
    x, W, experts = _setup()
    gamma = 1.0
    out, dropped = fwd(x, W, experts, 1, gamma)
    shape(out, (x.shape[0], experts[0].shape[1]), label="output")
    idx, _, _ = router(x, W, 1)
    cap = int(np.ceil(gamma * x.shape[0] * 1 / len(experts)))
    want = int(np.maximum(np.bincount(idx.ravel(), minlength=len(experts)) - cap, 0).sum())
    if want == 0:
        fail("test setup error: nothing overflows")
    equal(dropped, want, label="dropped assignments")


@check("a token dropped from its only choice has an all-zero row, and earlier rows get the seats")
def test_zero_rows(sub):
    fwd = has(sub, "moe_forward")
    x = np.array([[1.0], [1.0], [1.0], [1.0]])
    W = np.array([[5.0, 0.0]])  # every token picks expert 0
    experts = [np.array([[2.0]]), np.array([[3.0]])]
    out, dropped = fwd(x, W, experts, 1, 0.5)  # capacity = ceil(0.5 * 4 / 2) = 1
    equal(dropped, 3, label="dropped assignments")
    close(out[0], np.array([2.0]), atol=1e-9, label="row 0 (first in line)")
    close(out[1:], np.zeros((3, 1)), atol=1e-12, label="rows 1 to 3 (dropped)")


@check("with k = 2 a dropped slot is skipped and the other slot keeps its weight")
def test_partial_drop(sub):
    fwd = has(sub, "moe_forward")
    router = has(sub, "router")
    x = np.array([[1.0], [1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    experts = [np.array([[4.0]]), np.array([[10.0]]), np.array([[100.0]])]
    # capacity = max(1, ceil(0.25 * 2 * 2 / 3)) = 1: token 0 takes the only seats.
    out, dropped = fwd(x, W, experts, 2, 0.25)
    _, gates, _ = router(x, W, 2)
    equal(dropped, 2, label="dropped assignments")
    close(out[0], np.array([gates[0, 0] * 4.0 + gates[0, 1] * 10.0]), atol=1e-9, label="token 0")
    close(out[1], np.array([0.0]), atol=1e-12, label="token 1 lost both choices")


@check("a larger capacity factor never drops more")
def test_monotone(sub):
    fwd = has(sub, "moe_forward")
    x, W, experts = _setup(seed=5, n=40, E=4)
    last = None
    for gamma in [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]:
        _, dropped = fwd(x, W, experts, 2, gamma)
        if last is not None and dropped > last:
            fail(f"gamma={gamma} dropped {dropped}, more than the {last} dropped at the previous, smaller factor")
        last = dropped
    equal(last, 0, label="dropped at gamma = 4")
