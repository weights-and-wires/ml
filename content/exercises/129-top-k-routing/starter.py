"""Route each token to its top k experts.

Fill in the three functions below. `router` scores the experts and picks the
top k with renormalised weights, `moe_forward` mixes the chosen experts'
outputs, and `active_parameters` counts what is stored against what is run.
Each expert here is a single matrix, to keep the focus on the routing.
"""

import numpy as np


def router(
    x: np.ndarray, W_gate: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the top k experts for every token.

    Args:
        x: tokens, shape (n, d).
        W_gate: router weights, one column per expert, shape (d, E).
        k: how many experts each token uses, 1 <= k <= E.

    Returns:
        topk_indices: shape (n, k), expert numbers sorted from the highest
            logit to the lowest.
        gate_weights: shape (n, k), the softmax probabilities of the chosen
            experts divided by their own sum, so each row sums to one.
        full_probs: shape (n, E), the softmax over all E experts.

    Subtract each row's maximum logit before exponentiating, so that large
    logits do not overflow.
    """
    raise NotImplementedError("Implement router")


def moe_forward(
    x: np.ndarray, W_gate: np.ndarray, experts: list[np.ndarray], k: int
) -> np.ndarray:
    """Mix the top k experts for every token.

    Args:
        x: tokens, shape (n, d).
        W_gate: router weights, shape (d, E).
        experts: E matrices, each of shape (d, d_out).
        k: how many experts each token uses.

    Returns:
        Output of shape (n, d_out). Row t is the sum over its k chosen experts
        of gate_weight * (x[t] @ experts[expert]). Experts that a token did not
        choose must not contribute anything to that row.
    """
    raise NotImplementedError("Implement moe_forward")


def active_parameters(
    n_experts: int, k: int, expert_params: int, shared_params: int
) -> tuple[int, int]:
    """Count the parameters one token runs and the parameters the layer stores.

    Returns:
        (active, total). `active` counts k experts plus the shared parameters.
        `total` counts all n_experts experts plus the shared parameters.
    """
    raise NotImplementedError("Implement active_parameters")


if __name__ == "__main__":
    # The worked example: one token, three experts, scores [2, 1, 0], k = 2.
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
