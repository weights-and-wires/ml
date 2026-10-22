"""Reference solution: the load-balancing loss."""

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


if __name__ == "__main__":
    idx = np.array([[0], [0]])
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    print("f:", expert_load(idx, 2))
    print("P:", mean_gate_probability(probs))
    print("loss (expect 1.5):", load_balancing_loss(idx, probs, 2))

    even_idx = np.array([[0], [1], [2], [3]])
    even_probs = np.full((4, 4), 0.25)
    print("even routing (expect 1.0):", load_balancing_loss(even_idx, even_probs, 4))
