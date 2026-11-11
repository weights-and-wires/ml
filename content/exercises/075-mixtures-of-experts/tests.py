"""Checks for the mixture-of-experts exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _surrogate(x, W, k, outs, go):
    """The sum over tokens and chosen experts of gate * c, independent of the reader's code."""
    probs = _softmax(x @ W)
    total = 0.0
    for t in range(x.shape[0]):
        idx = np.argsort(-(x[t] @ W))[:k]
        g = probs[t, idx] / probs[t, idx].sum()
        total += float(g @ (outs[t, idx] @ go[t]))
    return total


def _setup(seed=0, n=12, d=4, E=3, d_out=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    W[:, 0] += 1.5  # expert 0 is popular, so it overflows
    experts = [rng.normal(size=(d, d_out)) for _ in range(E)]
    return x, W, experts


@check("router picks the true top-k logits largest first, with gate weights that sum to one and equal a softmax over the kept logits")
def test_router_topk(sub):
    router = has(sub, "router")
    rng = np.random.default_rng(0)
    n, d, E, k = 7, 5, 6, 3
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    idx, gates, probs = router(x, W, k)
    shape(idx, (n, k), label="topk_indices")
    shape(gates, (n, k), label="gate_weights")
    shape(probs, (n, E), label="full_probs")
    logits = x @ W
    for t in range(n):
        want = np.argsort(-logits[t])[:k]
        if list(idx[t]) != list(want):
            fail(f"token {t}: expected experts {list(want)} from largest logit to smallest, got {list(idx[t])}")
    close(gates.sum(axis=-1), np.ones(n), atol=1e-9, label="gate weight row sums")
    close(probs.sum(axis=-1), np.ones(n), atol=1e-9, label="full probability row sums")
    kept_logits = np.take_along_axis(logits, idx, axis=-1)
    close(gates, _softmax(kept_logits), atol=1e-9, label="gates vs softmax of kept logits")


@check("router gives weights 0.7311 and 0.2689 on the worked example, and stays finite when the logits are huge")
def test_router_worked(sub):
    router = has(sub, "router")
    x = np.array([[1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    idx, gates, probs = router(x, W, 2)
    close(probs[0], np.array([0.66524096, 0.24472847, 0.09003057]), atol=1e-6, label="full_probs")
    close(gates[0], np.array([0.73105858, 0.26894142]), atol=1e-6, label="gate_weights")
    x2 = np.array([[1.0], [1.0]])
    W2 = np.array([[1000.0, 999.0, -1000.0]])
    idx, gates, probs = router(x2, W2, 2)
    if not (np.all(np.isfinite(gates)) and np.all(np.isfinite(probs))):
        fail(f"got non-finite values: gates={gates}, probs={probs}. Subtract the row maximum before exp.")
    close(gates[0], np.array([0.73105858, 0.26894142]), atol=1e-6, label="gate_weights with huge logits")


@check("gate_logit_grad gives -0.3932 and +0.3932 on the worked example, sums to zero, and matches finite differences through the softmax")
def test_gate_grad(sub):
    f = has(sub, "gate_logit_grad")
    out = f(_softmax(np.array([1.0, 0.0])), np.array([1.0, 3.0]))
    close(out, np.array([-0.39322387, 0.39322387]), atol=1e-6, label="gate_logit_grad on the worked example")
    rng = np.random.default_rng(0)
    g = _softmax(rng.normal(size=4))
    c = rng.normal(size=4)
    close(np.sum(f(g, c)), 0.0, atol=1e-9, label="sum of the gradients")
    close(f(g, np.full(4, 2.5)), np.zeros(4), atol=1e-9, label="gradient when all c are equal")
    close(f(np.array([1.0]), np.array([3.0])), np.zeros(1), atol=1e-9, label="gradient when k = 1")
    rng = np.random.default_rng(1)
    z = rng.normal(size=3)
    c = rng.normal(size=3)
    fd = finite_difference(lambda zz: float(_softmax(zz) @ c), z)
    close(f(_softmax(z), c), fd, rtol=1e-5, atol=1e-6, label="gate_logit_grad vs finite differences")


@check("router_backward has shape (d, E), is zero when k = 1, and matches finite differences of the gate-weighted score")
def test_router_backward(sub):
    f = has(sub, "router_backward")
    rng = np.random.default_rng(2)
    n, d, E, d_out = 4, 3, 5, 2
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    outs = rng.normal(size=(n, E, d_out))
    go = rng.normal(size=(n, d_out))
    shape(f(x, W, 2, outs, go), (d, E), label="router_backward")
    close(f(x, W, 1, outs, go), np.zeros((d, E)), atol=1e-9, label="router_backward when k = 1")

    rng = np.random.default_rng(7)
    n, d, E, k, d_out = 6, 4, 4, 2, 3
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E)) * 1.5
    outs = rng.normal(size=(n, E, d_out))
    go = rng.normal(size=(n, d_out))
    fd = finite_difference(lambda w: _surrogate(x, w.reshape(d, E), k, outs, go), W.flatten()).reshape(d, E)
    close(f(x, W, k, outs, go), fd, rtol=1e-3, atol=1e-4, label="router_backward vs finite differences")


@check("expert_load counts slots and divides by n * k, so it sums to one")
def test_load(sub):
    f = has(sub, "expert_load")
    idx = np.array([[0, 1], [0, 2], [0, 1]])  # n = 3, k = 2, six slots
    got = f(idx, 4)
    shape(got, (4,), label="f")
    close(got, np.array([3 / 6, 2 / 6, 1 / 6, 0.0]), atol=1e-12, label="f")
    close(got.sum(), 1.0, atol=1e-12, label="sum of f")


@check("load_balancing_loss is 1.5 on the worked example, 1 for uniform routing, E for collapse with k = 1, and at most E / k")
def test_balancing_loss(sub):
    f = has(sub, "load_balancing_loss")
    close(f(np.array([[0], [0]]), np.array([[0.9, 0.1], [0.6, 0.4]]), 2), 1.5, atol=1e-12, label="worked example")
    E = 4
    close(f(np.array([[0], [1], [2], [3]]), np.full((4, E), 1 / E), E), 1.0, atol=1e-12, label="uniform, k = 1")
    idx2 = np.array([[0, 1], [2, 3], [0, 1], [2, 3]])
    close(f(idx2, np.full((4, E), 1 / E), E), 1.0, atol=1e-12, label="uniform, k = 2")
    n = 6
    one_hot = np.zeros((n, E))
    one_hot[:, 0] = 1.0
    close(f(np.zeros((n, 1), dtype=int), one_hot, E), 4.0, atol=1e-12, label="collapse, k = 1")
    two_hot = np.zeros((n, E))
    two_hot[:, :2] = 0.5
    close(f(np.tile(np.array([0, 1]), (n, 1)), two_hot, E), 2.0, atol=1e-12, label="collapse onto two experts, k = 2")
    rng = np.random.default_rng(3)
    E, n, k = 6, 40, 2
    for _ in range(5):
        probs = _softmax(rng.normal(size=(n, E)) * 2)
        idx = np.argsort(-probs, axis=1)[:, :k]
        loss = f(idx, probs, E)
        if loss > E / k + 1e-9 or loss < 0:
            fail(f"loss {loss:.4f} is outside [0, {E / k}] for k = {k}, E = {E}")


@check("expert_capacity gives 4 and 2 on the worked example, n*k for None, rounds up, and never returns less than 1")
def test_capacity(sub):
    f = has(sub, "expert_capacity")
    equal(f(6, 2, 3, 1.0), 4, label="capacity at gamma = 1")
    equal(f(6, 2, 3, 0.5), 2, label="capacity at gamma = 0.5")
    equal(f(6, 2, 3, None), 12, label="capacity when unlimited")
    equal(f(5, 1, 4, 1.0), 2, label="ceil(1.25)")
    equal(f(2, 1, 8, 0.5), 1, label="a tiny factor still leaves one seat")
    equal(f(8, 1, 4, 2.0), 4, label="eight tokens, four experts, gamma 2")


@check("moe_forward mixes the chosen experts: the hand mix 5.614, the dense sum when k = E, and no effect from unchosen experts")
def test_moe_mix(sub):
    fwd = has(sub, "moe_forward")
    x = np.array([[1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    experts = [np.array([[4.0]]), np.array([[10.0]]), np.array([[100.0]])]
    out, dropped = fwd(x, W, experts, 2)
    shape(out, (1, 1), label="output")
    close(out, np.array([[5.61364853]]), atol=1e-6, label="mixed output")
    equal(dropped, 0, label="dropped slots with no capacity limit")

    rng = np.random.default_rng(2)
    n, d, E, d_out = 6, 4, 4, 3
    xs = rng.normal(size=(n, d))
    Ws = rng.normal(size=(d, E))
    exps = [rng.normal(size=(d, d_out)) for _ in range(E)]
    probs = _softmax(xs @ Ws)
    want = sum(probs[:, [e]] * (xs @ exps[e]) for e in range(E))
    out, _ = fwd(xs, Ws, exps, E)
    shape(out, (n, d_out), label="dense output")
    close(out, want, atol=1e-9, label="dense weighted sum")

    a, _ = fwd(x, W, [np.array([[4.0]]), np.array([[10.0]]), np.array([[1.0]])], 2)
    b, _ = fwd(x, W, [np.array([[4.0]]), np.array([[10.0]]), np.array([[1e9]])], 2)
    close(a, b, atol=1e-9, label="output after changing an unchosen expert")


@check("moe_forward drops exactly the overflow, first come first served, keeps the surviving gate weight, and drops less as the capacity factor grows")
def test_moe_drops(sub):
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
    equal(dropped, want, label="dropped slots")

    xt = np.array([[1.0], [1.0], [1.0], [1.0]])
    Wt = np.array([[5.0, 0.0]])  # every token picks expert 0
    et = [np.array([[2.0]]), np.array([[3.0]])]
    out, dropped = fwd(xt, Wt, et, 1, 0.5)  # capacity = ceil(0.5 * 4 / 2) = 1
    equal(dropped, 3, label="dropped slots")
    close(out[0], np.array([2.0]), atol=1e-9, label="row 0 (first in line)")
    close(out[1:], np.zeros((3, 1)), atol=1e-12, label="rows 1 to 3 (dropped)")

    x2 = np.array([[1.0], [1.0]])
    W2 = np.array([[2.0, 1.0, 0.0]])
    e2 = [np.array([[4.0]]), np.array([[10.0]]), np.array([[100.0]])]
    # capacity = max(1, ceil(0.25 * 2 * 2 / 3)) = 1: token 0 takes the only seats.
    out, dropped = fwd(x2, W2, e2, 2, 0.25)
    _, gates, _ = router(x2, W2, 2)
    equal(dropped, 2, label="dropped slots with k = 2")
    close(out[0], np.array([gates[0, 0] * 4.0 + gates[0, 1] * 10.0]), atol=1e-9, label="token 0")
    close(out[1], np.array([0.0]), atol=1e-12, label="token 1 lost both choices")

    xm, Wm, em = _setup(seed=5, n=40, E=4)
    last = None
    for g in [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]:
        _, d = fwd(xm, Wm, em, 2, g)
        if last is not None and d > last:
            fail(f"gamma={g} dropped {d}, more than the {last} dropped at the previous, smaller factor")
        last = d
    equal(last, 0, label="dropped at gamma = 4")
