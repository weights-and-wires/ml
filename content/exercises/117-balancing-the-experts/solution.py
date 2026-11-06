"""Reference solution: the load-balancing loss, capacity limits, and dropping."""

import math

import numpy as np


def expert_load(expert_indices: np.ndarray, n_experts: int) -> np.ndarray:
    counts = np.bincount(expert_indices.ravel(), minlength=n_experts)
    return counts / expert_indices.size  # n * k slots in total


def mean_gate_probability(gate_probs: np.ndarray) -> np.ndarray:
    return gate_probs.mean(axis=0)


def load_balancing_loss(
    expert_indices: np.ndarray, gate_probs: np.ndarray, n_experts: int
) -> float:
    f = expert_load(expert_indices, n_experts)
    P = mean_gate_probability(gate_probs)
    return float(n_experts * np.sum(f * P))


# Written in the chapter "Send each token to a few experts".
def router(
    x: np.ndarray, W_gate: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (topk_indices, gate_weights, full_probs) for x of shape (n, d)."""
    logits = x @ W_gate
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    full_probs = exp / exp.sum(axis=-1, keepdims=True)
    topk_indices = np.argsort(-logits, axis=-1)[:, :k]
    kept = np.take_along_axis(full_probs, topk_indices, axis=-1)
    gate_weights = kept / kept.sum(axis=-1, keepdims=True)
    return topk_indices, gate_weights, full_probs


def expert_capacity(
    n_tokens: int, k: int, n_experts: int, capacity_factor: float | None
) -> int:
    """How many (token, slot) assignments one expert accepts per batch.

    Returns n_tokens * k when capacity_factor is None, so that nothing is ever
    dropped. Otherwise returns the ceiling of
    capacity_factor * n_tokens * k / n_experts, but never less than 1.
    """
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
    """Mix the top k experts, with a limit on how many assignments each takes.

    Args:
        x: tokens, shape (n, d).
        W_gate: shape (d, E).
        experts: E matrices of shape (d, d_out).
        k: experts per token.
        capacity_factor: passed to expert_capacity.

    Returns:
        (output, n_dropped). output has shape (n, d_out). Walk the tokens in
        row order and each token's slots in order. If the chosen expert has
        room, add gate_weight * (x[t] @ experts[expert]) to the token's row
        and use one seat. If it is full, add nothing and count one dropped
        assignment. A token dropped from every choice has an all-zero row.
    """
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


def dropped_fraction(n_dropped: int, n_tokens: int, k: int) -> float:
    """The share of all n_tokens * k assignments that were dropped."""
    return n_dropped / (n_tokens * k)


if __name__ == "__main__":
    idx = np.array([[0], [0]])
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    print("f:", expert_load(idx, 2))
    print("P:", mean_gate_probability(probs))
    print("loss (expect 1.5):", load_balancing_loss(idx, probs, 2))

    even_idx = np.array([[0], [1], [2], [3]])
    even_probs = np.full((4, 4), 0.25)
    print("even routing (expect 1.0):", load_balancing_loss(even_idx, even_probs, 4))

    # Capacity and dropping.
    print("capacity for n=6, k=2, E=3, gamma=1 (expect 4):", expert_capacity(6, 2, 3, 1.0))
    print("capacity with gamma=0.5 (expect 2):", expert_capacity(6, 2, 3, 0.5))

    rng = np.random.default_rng(0)
    n, d, E, k = 20, 6, 4, 2
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    W[:, 0] += 1.0  # make expert 0 popular
    experts = [rng.normal(size=(d, d)) for _ in range(E)]
    for gamma in [0.5, 1.0, 2.0, None]:
        _, dropped = moe_forward(x, W, experts, k, gamma)
        print(f"gamma={gamma}: dropped {dropped} of {n * k} ({dropped_fraction(dropped, n, k):.0%})")
