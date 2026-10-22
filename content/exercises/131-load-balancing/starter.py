"""Build the load-balancing loss.

Fill in three small functions. `expert_load` measures how the routing slots
were shared, `mean_gate_probability` averages the router's probabilities, and
`load_balancing_loss` combines them.
"""

import numpy as np


def expert_load(expert_indices: np.ndarray, n_experts: int) -> np.ndarray:
    """Return f, each expert's share of all routing slots.

    Args:
        expert_indices: shape (n, k), the experts each token chose.
        n_experts: E.

    Returns:
        Shape (E,). f[i] is the number of slots that chose expert i divided by
        n * k, the total number of slots. The entries sum to one.
    """
    raise NotImplementedError("Implement expert_load")


def mean_gate_probability(gate_probs: np.ndarray) -> np.ndarray:
    """Return P, the router's average probability for each expert.

    Args:
        gate_probs: shape (n, E), the full softmax for every token.

    Returns:
        Shape (E,), the mean over tokens. The entries sum to one.
    """
    raise NotImplementedError("Implement mean_gate_probability")


def load_balancing_loss(
    expert_indices: np.ndarray, gate_probs: np.ndarray, n_experts: int
) -> float:
    """Return E * sum_i f_i * P_i as a Python float.

    f counts the router's choices, before any capacity limit drops a token.
    """
    raise NotImplementedError("Implement load_balancing_loss")


if __name__ == "__main__":
    # The worked example: two tokens, two experts, k = 1, both sent to expert 0.
    idx = np.array([[0], [0]])
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    print("f:", expert_load(idx, 2))
    print("P:", mean_gate_probability(probs))
    print("loss (expect 1.5):", load_balancing_loss(idx, probs, 2))

    # Perfectly even routing over four experts scores exactly 1.
    even_idx = np.array([[0], [1], [2], [3]])
    even_probs = np.full((4, 4), 0.25)
    print("even routing (expect 1.0):", load_balancing_loss(even_idx, even_probs, 4))
