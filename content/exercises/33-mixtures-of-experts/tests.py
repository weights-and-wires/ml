"""Checks for the mixture-of-experts exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@check("router selects the true top-k logits per token")
def test_router_topk_validity(sub):
    got = has(sub, "router")
    rng = np.random.default_rng(0)
    n_tokens, d_model, n_experts, k = 7, 5, 6, 3
    x = rng.normal(size=(n_tokens, d_model))
    W_gate = rng.normal(size=(d_model, n_experts))

    topk_idx, gate_w, full_probs = got(x, W_gate, k)
    shape(topk_idx, (n_tokens, k), label="topk_indices")
    shape(gate_w, (n_tokens, k), label="gate_weights")
    shape(full_probs, (n_tokens, n_experts), label="full_probs")

    if topk_idx.min() < 0 or topk_idx.max() >= n_experts:
        fail(f"topk_indices must lie in [0, {n_experts}), got range "
             f"[{topk_idx.min()}, {topk_idx.max()}]")

    logits = x @ W_gate
    for t in range(n_tokens):
        want = set(np.argsort(-logits[t])[:k].tolist())
        got_set = set(topk_idx[t].tolist())
        if got_set != want:
            fail(f"token {t}: top-{k} experts should be {want}, got {got_set}")


@check("gate weights are non-negative and sum to one per token")
def test_gate_weights_normalise(sub):
    got = has(sub, "router")
    rng = np.random.default_rng(1)
    n_tokens, d_model, n_experts, k = 10, 4, 5, 2
    x = rng.normal(size=(n_tokens, d_model))
    W_gate = rng.normal(size=(d_model, n_experts))

    _, gate_w, full_probs = got(x, W_gate, k)
    if np.any(gate_w < -1e-10):
        fail(f"gate_weights has a negative entry: min is {gate_w.min():.3g}")
    close(gate_w.sum(axis=-1), np.ones(n_tokens), atol=1e-8, label="gate_weights row sums")
    close(full_probs.sum(axis=-1), np.ones(n_tokens), atol=1e-8, label="full_probs row sums")


@check("moe_forward with k = n_experts and no capacity limit is a dense weighted combination")
def test_moe_forward_dense_equivalence(sub):
    router = has(sub, "router")
    got = has(sub, "moe_forward")
    rng = np.random.default_rng(2)
    n_tokens, d_model, n_experts = 8, 4, 4
    x = rng.normal(size=(n_tokens, d_model))
    W_gate = rng.normal(size=(d_model, n_experts))
    experts = [rng.normal(size=(d_model, d_model)) for _ in range(n_experts)]

    output, n_dropped = got(x, W_gate, experts, n_experts, None)
    if n_dropped != 0:
        fail(f"an unlimited capacity must never drop a token, but {n_dropped} were dropped")

    _, _, full_probs = router(x, W_gate, n_experts)
    dense = sum(full_probs[:, [i]] * (x @ experts[i]) for i in range(n_experts))
    close(output, dense, rtol=1e-8, atol=1e-9,
          label="moe_forward against the dense weighted combination")


@check("load_balancing_loss matches a hand computation of f and P")
def test_load_balancing_loss_matches_manual(sub):
    got = has(sub, "load_balancing_loss")
    rng = np.random.default_rng(3)
    n_tokens, n_experts, k = 12, 5, 2
    expert_indices = rng.integers(0, n_experts, size=(n_tokens, k))
    logits = rng.normal(size=(n_tokens, n_experts))
    gate_probs = _softmax(logits)

    counts = np.bincount(expert_indices.ravel(), minlength=n_experts)
    f = counts / (n_tokens * k)
    P = gate_probs.mean(axis=0)
    want = float(n_experts * np.sum(f * P))

    value = got(expert_indices, gate_probs, n_experts)
    if abs(value - want) > 1e-8 * max(1.0, abs(want)):
        fail(f"load_balancing_loss gave {value:.6g}, expected {want:.6g}")


@check("the auxiliary loss is minimised by uniform routing and maximised by collapse")
def test_load_balancing_loss_extremes(sub):
    got = has(sub, "load_balancing_loss")
    n_experts, k = 4, 1
    n_tokens = 400  # a multiple of n_experts, so uniform routing is exact

    # Perfectly uniform: every expert gets the same share of assignments and
    # the same mean gate probability.
    uniform_indices = np.tile(np.arange(n_experts), n_tokens // n_experts).reshape(-1, 1)
    uniform_probs = np.full((n_tokens, n_experts), 1.0 / n_experts)
    uniform_loss = got(uniform_indices, uniform_probs, n_experts)
    if abs(uniform_loss - 1.0) > 1e-6:
        fail(f"uniform routing should give a loss of exactly 1.0 (alpha=1), got {uniform_loss:.6g}")

    # Fully collapsed: every token routes to expert 0 with near-certainty.
    collapsed_indices = np.zeros((n_tokens, 1), dtype=int)
    collapsed_probs = np.zeros((n_tokens, n_experts))
    collapsed_probs[:, 0] = 1.0
    collapsed_loss = got(collapsed_indices, collapsed_probs, n_experts)
    if abs(collapsed_loss - n_experts) > 1e-6:
        fail(f"total collapse should give a loss of exactly N={n_experts}, got {collapsed_loss:.6g}")

    # A handful of random routings should land strictly between the two.
    rng = np.random.default_rng(4)
    for _ in range(5):
        logits = rng.normal(size=(n_tokens, n_experts)) * rng.uniform(0.5, 3.0)
        probs = _softmax(logits)
        indices = np.argmax(probs, axis=-1, keepdims=True)
        random_loss = got(indices, probs, n_experts)
        if not (uniform_loss - 1e-6 <= random_loss <= collapsed_loss + 1e-6):
            fail(
                f"a random routing gave a loss of {random_loss:.6g}, outside "
                f"[{uniform_loss:.6g}, {collapsed_loss:.6g}]"
            )


@check("raising the capacity factor monotonically reduces dropped assignments")
def test_capacity_monotonic(sub):
    got = has(sub, "moe_forward")
    rng = np.random.default_rng(5)
    n_tokens, d_model, n_experts, k = 30, 4, 4, 2
    x = rng.normal(size=(n_tokens, d_model))
    W_gate = rng.normal(size=(d_model, n_experts))
    experts = [rng.normal(size=(d_model, d_model)) for _ in range(n_experts)]

    factors = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
    dropped_counts = []
    for factor in factors:
        _, n_dropped = got(x, W_gate, experts, k, factor)
        dropped_counts.append(n_dropped)

    for earlier, later in zip(dropped_counts, dropped_counts[1:]):
        if later > earlier:
            fail(f"dropped counts {dropped_counts} are not monotonically non-increasing "
                 f"as the capacity factor grows over {factors}")
    if dropped_counts[-1] != 0:
        fail(f"a capacity factor of {factors[-1]} should leave nothing dropped, "
             f"but {dropped_counts[-1]} assignments were")


@check("a token dropped from its only expert contributes nothing to the output")
def test_dropped_token_is_zero(sub):
    got = has(sub, "moe_forward")
    rng = np.random.default_rng(6)
    n_tokens, d_model, n_experts, k = 6, 3, 3, 1

    # Column 0 of x is positive for every token, and W_gate is built so expert
    # 0 wins by a wide margin whenever x[:, 0] > 0 -- every token routes there.
    x = rng.uniform(0.1, 1.0, size=(n_tokens, d_model))
    W_gate = np.zeros((d_model, n_experts))
    W_gate[0, 0] = 100.0
    W_gate[0, 1:] = -100.0
    experts = [rng.normal(size=(d_model, d_model)) for _ in range(n_experts)]

    # Capacity 1 for expert 0 means only the first token in row order keeps
    # its assignment; every other token is dropped from its only slot.
    output, n_dropped = got(x, W_gate, experts, k, capacity_factor=0.5)
    # capacity = ceil(0.5 * 6 * 1 / 3) = 1
    if n_dropped != n_tokens - 1:
        fail(f"expected {n_tokens - 1} dropped assignments with capacity 1, got {n_dropped}")

    close(output[1:], np.zeros((n_tokens - 1, d_model)), atol=1e-12,
          label="output rows for tokens dropped from their only expert")
    if np.allclose(output[0], 0.0):
        fail("the surviving token's output should not be zero")


@check("active_parameters matches a hand-computed value")
def test_active_parameters(sub):
    got = has(sub, "active_parameters")
    active, total = got(n_experts=8, k=2, expert_params=100_000_000, shared_params=100_000_000)
    equal(active, 300_000_000, label="active parameters")
    equal(total, 900_000_000, label="total parameters")

    # k = n_experts must recover a fully dense layer: active equals total.
    active_dense, total_dense = got(n_experts=5, k=5, expert_params=40, shared_params=0)
    equal(active_dense, total_dense, label="active parameters when k = n_experts")


@check("router_backward matches mlcheck.finite_difference")
def test_router_backward(sub):
    router = has(sub, "router")
    got = has(sub, "router_backward")

    rng = np.random.default_rng(7)
    n_tokens, d_model, n_experts, k, d_out = 6, 4, 4, 2, 3
    x = rng.normal(size=(n_tokens, d_model))
    W_gate = rng.normal(size=(d_model, n_experts)) * 1.5
    expert_outputs = rng.normal(size=(n_tokens, n_experts, d_out))
    grad_output = rng.normal(size=(n_tokens, d_out))

    def scalar_loss(w_flat):
        W = w_flat.reshape(d_model, n_experts)
        topk_idx, gate_w, _ = router(x, W, k)
        total = 0.0
        for t in range(n_tokens):
            for slot, e in enumerate(topk_idx[t]):
                total += gate_w[t, slot] * float(grad_output[t] @ expert_outputs[t, e])
        return total

    fd = finite_difference(scalar_loss, W_gate.flatten()).reshape(d_model, n_experts)
    analytic = got(x, W_gate, k, expert_outputs, grad_output)
    shape(analytic, (d_model, n_experts), label="router_backward output")
    close(analytic, fd, rtol=1e-3, atol=1e-4, label="router_backward vs finite differences")
