"""Reference solution for the mixture-of-experts exercise."""

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def router(
    x: np.ndarray, W_gate: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = x @ W_gate
    full_probs = _softmax(logits)

    # Sort experts by logit, descending; the top k are the first k columns.
    order = np.argsort(-logits, axis=-1)
    topk_indices = order[:, :k]

    topk_probs = np.take_along_axis(full_probs, topk_indices, axis=-1)
    gate_weights = topk_probs / np.sum(topk_probs, axis=-1, keepdims=True)

    return topk_indices, gate_weights, full_probs


def load_balancing_loss(
    expert_indices: np.ndarray, gate_probs: np.ndarray, n_experts: int
) -> float:
    n_tokens, k = expert_indices.shape
    counts = np.bincount(expert_indices.ravel(), minlength=n_experts)
    f = counts / (n_tokens * k)
    P = gate_probs.mean(axis=0)
    return float(n_experts * np.sum(f * P))


def moe_forward(
    x: np.ndarray,
    W_gate: np.ndarray,
    experts: list[np.ndarray],
    k: int,
    capacity_factor: float | None = None,
) -> tuple[np.ndarray, int]:
    n_tokens = x.shape[0]
    n_experts = len(experts)
    d_out = experts[0].shape[1]

    topk_indices, gate_weights, _ = router(x, W_gate, k)

    if capacity_factor is None:
        capacity = n_tokens * k  # an assignment count no expert can exceed
    else:
        capacity = max(1, int(np.ceil(capacity_factor * n_tokens * k / n_experts)))

    counts = np.zeros(n_experts, dtype=int)
    output = np.zeros((n_tokens, d_out))
    n_dropped = 0

    for t in range(n_tokens):
        for slot in range(k):
            e = int(topk_indices[t, slot])
            if counts[e] < capacity:
                counts[e] += 1
                output[t] += gate_weights[t, slot] * (x[t] @ experts[e])
            else:
                n_dropped += 1

    return output, n_dropped


def active_parameters(
    n_experts: int, k: int, expert_params: int, shared_params: int
) -> tuple[int, int]:
    active = k * expert_params + shared_params
    total = n_experts * expert_params + shared_params
    return active, total


def router_backward(
    x: np.ndarray,
    W_gate: np.ndarray,
    k: int,
    expert_outputs: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    topk_indices, gate_weights, full_probs = router(x, W_gate, k)
    n_tokens, n_experts = full_probs.shape

    dz = np.zeros((n_tokens, n_experts))
    for t in range(n_tokens):
        idx = topk_indices[t]
        g = gate_weights[t]
        c = np.array([grad_output[t] @ expert_outputs[t, e] for e in idx])
        c_bar = float(np.dot(g, c))
        dz[t, idx] = g * (c - c_bar)

    return x.T @ dz


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_tokens, d_model, n_experts, k = 20, 8, 4, 2

    x = rng.normal(size=(n_tokens, d_model))
    W_gate = rng.normal(size=(d_model, n_experts)) * 0.5
    experts = [rng.normal(size=(d_model, d_model)) * 0.3 for _ in range(n_experts)]

    topk_idx, gate_w, full_probs = router(x, W_gate, k)
    print(f"gate weight row sums: {gate_w.sum(axis=-1)[:5]}")

    aux_loss = load_balancing_loss(topk_idx, full_probs, n_experts)
    print(f"load balancing loss: {aux_loss:.4f} (uniform would be {1.0:.4f})")

    for capacity_factor in [0.5, 1.0, 2.0, None]:
        output, n_dropped = moe_forward(x, W_gate, experts, k, capacity_factor)
        label = "unlimited" if capacity_factor is None else str(capacity_factor)
        print(f"capacity_factor={label:>9}: dropped {n_dropped} assignments")

    active, total = active_parameters(n_experts=8, k=2, expert_params=100_000_000, shared_params=100_000_000)
    print(f"active params: {active:,}, total params: {total:,}, ratio: {active / total:.3f}")
