"""Mixtures of experts: route tokens, train the router, balance the load.

A router scores every expert for each token, keeps the top k, and mixes only
those experts with weights that sum to one. The router learns through those
weights, because the top-k choice itself has no slope. An auxiliary loss keeps
the traffic even, and a capacity limit drops the slots that overflow.

An expert here is a single matrix of shape (d, d_out), so its output for a
token x is x @ expert. A "slot" is one (token, expert) choice, so a batch of n
tokens holds n * k slots.
"""

import math

import numpy as np


def router(
    x: np.ndarray, W_gate: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every expert for each token and keep the top k.

    Args:
        x: tokens, shape (n, d).
        W_gate: the router matrix, shape (d, E).
        k: how many experts each token visits.

    Returns:
        A tuple (topk_indices, gate_weights, full_probs):
        topk_indices has shape (n, k), the chosen experts from the largest
        logit to the smallest; gate_weights has shape (n, k), the softmax
        probabilities of the chosen experts divided by their own sum, so each
        row sums to one; full_probs has shape (n, E), the softmax over all
        experts. The softmax must stay finite when the logits are huge.
    """
    raise NotImplementedError("router")


def gate_logit_grad(g: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Gradient of the loss on the logits of one token's chosen experts.

    Args:
        g: the token's gate weights, shape (k,), summing to one.
        c: shape (k,). c[i] is the loss slope per unit of weight on chosen
            expert i, that is, grad_output dotted with that expert's output.

    Returns:
        Shape (k,): g[j] * (c[j] - c_bar), where c_bar is the gate-weighted
        average of c.
    """
    raise NotImplementedError("gate_logit_grad")


def router_backward(
    x: np.ndarray,
    W_gate: np.ndarray,
    k: int,
    expert_outputs: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Gradient of the router's gate-weighted score with respect to W_gate.

    Args:
        x: tokens, shape (n, d).
        W_gate: shape (d, E).
        k: experts per token.
        expert_outputs: shape (n, E, d_out). Entry [t, e] is expert e's
            output for token t. Treat it as fixed.
        grad_output: shape (n, d_out), the loss gradient with respect to each
            token's mixed output. Treat it as fixed.

    Returns:
        Shape (d, E). Build dz of shape (n, E), zero everywhere except at each
        token's chosen experts, where it holds gate_logit_grad. Return
        x.T @ dz.
    """
    raise NotImplementedError("router_backward")


def expert_load(expert_indices: np.ndarray, n_experts: int) -> np.ndarray:
    """The share of all slots that went to each expert.

    Args:
        expert_indices: integer array of shape (n, k).
        n_experts: the number of experts E.

    Returns:
        f of shape (n_experts,): each expert's slot count divided by n * k, so
        it sums to one.
    """
    raise NotImplementedError("expert_load")


def load_balancing_loss(
    expert_indices: np.ndarray, gate_probs: np.ndarray, n_experts: int
) -> float:
    """The auxiliary loss E * sum_i f_i * P_i.

    Args:
        expert_indices: chosen experts, shape (n, k).
        gate_probs: the softmax over all experts, shape (n, E).
        n_experts: the number of experts E.

    Returns:
        A float. f is expert_load, and P is the mean of gate_probs over
        tokens. It equals 1 for perfectly even routing.
    """
    raise NotImplementedError("load_balancing_loss")


def expert_capacity(
    n_tokens: int, k: int, n_experts: int, capacity_factor: float | None
) -> int:
    """How many slots one expert accepts per batch.

    Returns:
        n_tokens * k when capacity_factor is None, so nothing is ever dropped.
        Otherwise the ceiling of capacity_factor * n_tokens * k / n_experts,
        but never less than 1.
    """
    raise NotImplementedError("expert_capacity")


def moe_forward(
    x: np.ndarray,
    W_gate: np.ndarray,
    experts: list[np.ndarray],
    k: int,
    capacity_factor: float | None = None,
) -> tuple[np.ndarray, int]:
    """Mix the top k experts, with a limit on how many slots each takes.

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
        slot. The surviving gate weight is not rescaled, and a token dropped
        from every choice has an all-zero row.
    """
    raise NotImplementedError("moe_forward")


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
