"""Checks for the capacity and dropping exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def _setup(seed=0, n=12, d=4, E=3, d_out=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    W[:, 0] += 1.5  # expert 0 is popular, so it overflows
    experts = [rng.normal(size=(d, d_out)) for _ in range(E)]
    return x, W, experts


@check("expert_capacity gives 4 and 2 on the worked example, and n*k for None")
def test_capacity(sub):
    f = has(sub, "expert_capacity")
    equal(f(6, 2, 3, 1.0), 4, label="capacity at gamma = 1")
    equal(f(6, 2, 3, 0.5), 2, label="capacity at gamma = 0.5")
    equal(f(6, 2, 3, None), 12, label="capacity when unlimited")


@check("expert_capacity rounds up, and never returns less than 1")
def test_capacity_edges(sub):
    f = has(sub, "expert_capacity")
    equal(f(5, 1, 4, 1.0), 2, label="ceil(1.25)")
    equal(f(2, 1, 8, 0.5), 1, label="a tiny factor still leaves one seat")
    equal(f(8, 1, 4, 2.0), 4, label="eight tokens, four experts, gamma 2")


@check("dropped_fraction divides by all n * k assignments")
def test_fraction(sub):
    f = has(sub, "dropped_fraction")
    close(f(6, 12, 2), 0.25, atol=1e-12, label="6 of 24")
    close(f(0, 10, 3), 0.0, atol=1e-12, label="nothing dropped")


@check("moe_forward with no limit matches the plain mix and drops nothing")
def test_unlimited(sub):
    fwd = has(sub, "moe_forward")
    router = has(sub, "router")
    x, W, experts = _setup()
    out, dropped = fwd(x, W, experts, 2, None)
    equal(dropped, 0, label="dropped assignments")
    idx, gates, _ = router(x, W, 2)
    want = np.zeros((x.shape[0], experts[0].shape[1]))
    for t in range(x.shape[0]):
        for s in range(2):
            want[t] += gates[t, s] * (x[t] @ experts[idx[t, s]])
    close(out, want, atol=1e-9, label="unlimited output")


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
