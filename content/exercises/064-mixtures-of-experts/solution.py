"""Reference solution: route tokens to experts, train the router, balance the load."""

import math

import numpy as np


def router(
    x: np.ndarray, W_gate: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = x @ W_gate
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    full_probs = exp / exp.sum(axis=-1, keepdims=True)

    # argsort of the negated logits lists the experts from largest to smallest.
    topk_indices = np.argsort(-logits, axis=-1)[:, :k]
    kept = np.take_along_axis(full_probs, topk_indices, axis=-1)
    gate_weights = kept / kept.sum(axis=-1, keepdims=True)
    return topk_indices, gate_weights, full_probs


def gate_logit_grad(g: np.ndarray, c: np.ndarray) -> np.ndarray:
    c_bar = np.sum(g * c)
    return g * (c - c_bar)


def router_backward(
    x: np.ndarray,
    W_gate: np.ndarray,
    k: int,
    expert_outputs: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    topk_indices, gate_weights, _ = router(x, W_gate, k)
    dz = np.zeros((x.shape[0], W_gate.shape[1]))
    for t in range(x.shape[0]):
        chosen = topk_indices[t]
        c = expert_outputs[t, chosen] @ grad_output[t]
        dz[t, chosen] = gate_logit_grad(gate_weights[t], c)
    return x.T @ dz


def expert_load(expert_indices: np.ndarray, n_experts: int) -> np.ndarray:
    counts = np.bincount(expert_indices.ravel(), minlength=n_experts)
    return counts / expert_indices.size  # n * k slots in total


def load_balancing_loss(
    expert_indices: np.ndarray, gate_probs: np.ndarray, n_experts: int
) -> float:
    f = expert_load(expert_indices, n_experts)
    P = gate_probs.mean(axis=0)
    return float(n_experts * np.sum(f * P))


def expert_capacity(
    n_tokens: int, k: int, n_experts: int, capacity_factor: float | None
) -> int:
    if capacity_factor is None:
        return n_tokens * k
    # The small subtraction guards against a float like 4.000000000000001.
    return max(1, math.ceil(capacity_factor * n_tokens * k / n_experts - 1e-9))


def moe_forward(
    x: np.ndarray,
    W_gate: np.ndarray,
    experts: list[np.ndarray],
    k: int,
    capacity_factor: float | None = None,
) -> tuple[np.ndarray, int]:
    n_tokens = x.shape[0]
    capacity = expert_capacity(n_tokens, k, len(experts), capacity_factor)
    topk_indices, gate_weights, _ = router(x, W_gate, k)

    output = np.zeros((n_tokens, experts[0].shape[1]))
    seats_used = np.zeros(len(experts), dtype=int)
    n_dropped = 0
    for t in range(n_tokens):
        for slot in range(k):
            expert = int(topk_indices[t, slot])
            if seats_used[expert] < capacity:
                seats_used[expert] += 1
                output[t] += gate_weights[t, slot] * (x[t] @ experts[expert])
            else:
                n_dropped += 1
    return output, n_dropped


if __name__ == "__main__":
    # Route one token to two of three experts.
    x = np.array([[1.0]])
    W_gate = np.array([[2.0, 1.0, 0.0]])
    experts = [np.array([[4.0]]), np.array([[10.0]]), np.array([[100.0]])]
    idx, gates, probs = router(x, W_gate, 2)
    print("chosen experts:", idx[0])
    print("gate weights (expect 0.7311, 0.2689):", np.round(gates[0], 4))
    out, dropped = moe_forward(x, W_gate, experts, 2)
    print("mixed output (expect 5.614):", np.round(out[0], 3), "dropped:", dropped)

    # The router's gradient through the gate weights.
    g, c = np.array([0.7311, 0.2689]), np.array([1.0, 3.0])
    print("logit gradients (expect -0.3932, 0.3932):", np.round(gate_logit_grad(g, c), 4))

    # The balancing loss and the capacity limit.
    idx2 = np.array([[0], [0]])
    probs2 = np.array([[0.9, 0.1], [0.6, 0.4]])
    print("balancing loss (expect 1.5):", load_balancing_loss(idx2, probs2, 2))
    print("capacity for n=6, k=2, E=3, gamma=1 (expect 4):", expert_capacity(6, 2, 3, 1.0))

    rng = np.random.default_rng(0)
    n, d, E, k = 20, 6, 4, 2
    xs = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    W[:, 0] += 1.0  # make expert 0 popular
    ex = [rng.normal(size=(d, d)) for _ in range(E)]
    for gamma in [0.5, 1.0, 2.0, None]:
        _, dropped = moe_forward(xs, W, ex, k, gamma)
        print(f"gamma={gamma}: dropped {dropped} of {n * k} ({dropped / (n * k):.0%})")
