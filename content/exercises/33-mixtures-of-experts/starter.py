"""Route, balance, and drop: the mechanics of a sparse mixture of experts.

Fill in the five functions below. `router` and `moe_forward` build the forward
pass this lesson derives; `load_balancing_loss` is the auxiliary loss that
keeps routing from collapsing; `active_parameters` is the parameter count the
sparsity is supposed to buy you; `router_backward` is the router's own
gradient, computed by hand rather than by an autodiff engine.

Each expert is represented as a plain weight matrix: expert i's output on a
token x is `x @ experts[i]`, so a "feed-forward network" here is a single
linear map. That keeps every check about the routing logic, not about what
happens to sit inside an expert.
"""

import numpy as np


def router(
    x: np.ndarray, W_gate: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k router with a softmax gate.

    Args:
        x: Token representations, shape (n_tokens, d_model).
        W_gate: Router weights, shape (d_model, n_experts).
        k: Number of experts to keep per token.

    Returns:
        A tuple (topk_indices, gate_weights, full_probs):
          - topk_indices: shape (n_tokens, k), the indices of the k experts
            with the largest logits for each token, sorted from largest to
            smallest.
          - gate_weights: shape (n_tokens, k), the softmax probabilities at
            those indices, renormalised so each row sums to 1.
          - full_probs: shape (n_tokens, n_experts), the softmax over all
            experts before the top-k cutoff. Each row sums to 1.
    """
    raise NotImplementedError("router")


def load_balancing_loss(
    expert_indices: np.ndarray, gate_probs: np.ndarray, n_experts: int
) -> float:
    """The auxiliary load-balancing loss, with alpha = 1.

    Args:
        expert_indices: The experts selected per token, shape (n_tokens, k),
            integer values in [0, n_experts).
        gate_probs: The full softmax probabilities before any top-k cutoff,
            shape (n_tokens, n_experts). Each row sums to 1.
        n_experts: The total number of experts, N.

    Returns:
        N * sum_i f_i * P_i, where f_i is the fraction of (token, slot)
        assignments routed to expert i, and P_i is the mean of gate_probs[:, i]
        over tokens.
    """
    raise NotImplementedError("load_balancing_loss")


def moe_forward(
    x: np.ndarray,
    W_gate: np.ndarray,
    experts: list[np.ndarray],
    k: int,
    capacity_factor: float | None = None,
) -> tuple[np.ndarray, int]:
    """The full sparse MoE forward pass, with capacity-limited dispatch.

    Tokens are processed in row order. For each of a token's k selected
    experts, add that expert's contribution if the expert has not yet reached
    its capacity for this batch; otherwise drop that assignment and add
    nothing for it.

    Args:
        x: Token representations, shape (n_tokens, d_model).
        W_gate: Router weights, shape (d_model, n_experts).
        experts: A list of n_experts weight matrices, each shape
            (d_model, d_out). Expert i's output on token x is x @ experts[i].
        k: Number of experts to keep per token.
        capacity_factor: If None, every expert has unlimited capacity and no
            assignment is ever dropped. Otherwise, each expert's capacity is
            ceil(capacity_factor * n_tokens * k / n_experts).

    Returns:
        A tuple (output, n_dropped):
          - output: shape (n_tokens, d_out), the gate-weighted sum of the
            surviving expert contributions for each token.
          - n_dropped: the total number of (token, slot) assignments dropped
            because their expert was already at capacity.
    """
    raise NotImplementedError("moe_forward")


def active_parameters(
    n_experts: int, k: int, expert_params: int, shared_params: int
) -> tuple[int, int]:
    """The active and total parameter counts of an MoE feed-forward layer.

    Args:
        n_experts: The total number of routed experts, N.
        k: The number of experts a token is routed to.
        expert_params: The parameter count of a single expert.
        shared_params: The combined parameter count of any always-on
            (shared) experts applied to every token.

    Returns:
        A tuple (active, total):
          - active: k * expert_params + shared_params.
          - total: n_experts * expert_params + shared_params.
    """
    raise NotImplementedError("active_parameters")


def router_backward(
    x: np.ndarray,
    W_gate: np.ndarray,
    k: int,
    expert_outputs: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """The router's gradient with respect to W_gate.

    For each token, with gate weights g over its selected experts and
    c_i = grad_output . expert_outputs[i] for each selected expert i, the
    gradient into that token's pre-softmax logit for expert i is
    g_i * (c_i - sum_j g_j * c_j), and 0 for every expert the token did not
    select.

    Args:
        x: Token representations, shape (n_tokens, d_model).
        W_gate: Router weights, shape (d_model, n_experts).
        k: Number of experts kept per token.
        expert_outputs: E_i(x_t) for every expert and every token, shape
            (n_tokens, n_experts, d_out).
        grad_output: The upstream gradient dL/dy, shape (n_tokens, d_out).

    Returns:
        dL/dW_gate, shape (d_model, n_experts).
    """
    raise NotImplementedError("router_backward")


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
