"""Checks for the top-k routing exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@check("router picks the true top-k logits, largest first")
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


@check("the worked example gives weights 0.7311 and 0.2689 from probabilities 0.665, 0.245, 0.090")
def test_router_worked_example(sub):
    router = has(sub, "router")
    x = np.array([[1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    idx, gates, probs = router(x, W, 2)
    close(probs[0], np.array([0.66524096, 0.24472847, 0.09003057]), atol=1e-6, label="full_probs")
    close(gates[0], np.array([0.73105858, 0.26894142]), atol=1e-6, label="gate_weights")


@check("gate weights sum to one and equal a softmax over only the kept logits")
def test_router_gates(sub):
    router = has(sub, "router")
    rng = np.random.default_rng(1)
    n, d, E, k = 10, 4, 5, 2
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    idx, gates, probs = router(x, W, k)
    close(gates.sum(axis=-1), np.ones(n), atol=1e-9, label="gate weight row sums")
    close(probs.sum(axis=-1), np.ones(n), atol=1e-9, label="full probability row sums")
    kept_logits = np.take_along_axis(x @ W, idx, axis=-1)
    close(gates, _softmax(kept_logits), atol=1e-9, label="gates vs softmax of kept logits")


@check("router stays finite when the logits are huge")
def test_router_stability(sub):
    router = has(sub, "router")
    x = np.array([[1.0], [1.0]])
    W = np.array([[1000.0, 999.0, -1000.0]])
    idx, gates, probs = router(x, W, 2)
    if not (np.all(np.isfinite(gates)) and np.all(np.isfinite(probs))):
        fail(f"got non-finite values: gates={gates}, probs={probs}. Subtract the row maximum before exp.")
    close(gates[0], np.array([0.73105858, 0.26894142]), atol=1e-6, label="gate_weights")


@check("moe_forward reproduces the hand-computed mix 5.614")
def test_moe_hand(sub):
    fwd = has(sub, "moe_forward")
    x = np.array([[1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    experts = [np.array([[4.0]]), np.array([[10.0]]), np.array([[100.0]])]
    out = fwd(x, W, experts, 2)
    shape(out, (1, 1), label="output")
    close(out, np.array([[5.61364853]]), atol=1e-6, label="mixed output")


@check("moe_forward with k = E is the dense weighted sum of every expert")
def test_moe_dense(sub):
    fwd = has(sub, "moe_forward")
    rng = np.random.default_rng(2)
    n, d, E, d_out = 6, 4, 4, 3
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    experts = [rng.normal(size=(d, d_out)) for _ in range(E)]
    probs = _softmax(x @ W)
    want = sum(probs[:, [e]] * (x @ experts[e]) for e in range(E))
    out = fwd(x, W, experts, E)
    shape(out, (n, d_out), label="output")
    close(out, want, atol=1e-9, label="dense weighted sum")


@check("moe_forward ignores experts a token did not choose")
def test_moe_unchosen(sub):
    fwd = has(sub, "moe_forward")
    x = np.array([[1.0]])
    W = np.array([[2.0, 1.0, 0.0]])
    a = fwd(x, W, [np.array([[4.0]]), np.array([[10.0]]), np.array([[1.0]])], 2)
    b = fwd(x, W, [np.array([[4.0]]), np.array([[10.0]]), np.array([[1e9]])], 2)
    close(a, b, atol=1e-9, label="output after changing an unchosen expert")


@check("active_parameters gives 300M run and 900M stored, and matches when k = E")
def test_active(sub):
    got = has(sub, "active_parameters")
    active, total = got(n_experts=8, k=2, expert_params=100_000_000, shared_params=100_000_000)
    equal(active, 300_000_000, label="active parameters")
    equal(total, 900_000_000, label="total parameters")
    active, total = got(n_experts=5, k=5, expert_params=40, shared_params=7)
    equal(active, total, label="active vs total when k = E")
