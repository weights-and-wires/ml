"""Reference solution: route each token to its top k experts."""

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


def moe_forward(
    x: np.ndarray, W_gate: np.ndarray, experts: list[np.ndarray], k: int
) -> np.ndarray:
    topk_indices, gate_weights, _ = router(x, W_gate, k)
    output = np.zeros((x.shape[0], experts[0].shape[1]))
    for t in range(x.shape[0]):
        for slot in range(k):
            expert = int(topk_indices[t, slot])
            output[t] += gate_weights[t, slot] * (x[t] @ experts[expert])
    return output


def active_parameters(
    n_experts: int, k: int, expert_params: int, shared_params: int
) -> tuple[int, int]:
    active = k * expert_params + shared_params
    total = n_experts * expert_params + shared_params
    return active, total


if __name__ == "__main__":
    x = np.array([[1.0]])
    W_gate = np.array([[2.0, 1.0, 0.0]])
    experts = [np.array([[4.0]]), np.array([[10.0]]), np.array([[100.0]])]

    idx, gates, probs = router(x, W_gate, 2)
    print("chosen experts:", idx[0])
    print("full probabilities:", np.round(probs[0], 4))
    print("gate weights:", np.round(gates[0], 4))
    print("mixed output (expect 5.614):", np.round(moe_forward(x, W_gate, experts, 2)[0], 3))

    active, total = active_parameters(8, 2, 100_000_000, 100_000_000)
    print(f"active {active:,} of {total:,} stored ({active / total:.0%})")
