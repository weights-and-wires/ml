"""Reference solution: train the router through the gate weights."""

import numpy as np


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
    c_bar = np.sum(g * c)
    return g * (c - c_bar)


def surrogate_loss(
    x: np.ndarray,
    W_gate: np.ndarray,
    k: int,
    expert_outputs: np.ndarray,
    grad_output: np.ndarray,
) -> float:
    """The scalar whose gradient with respect to W_gate is router_backward.

    Args:
        x: tokens, shape (n, d).
        W_gate: shape (d, E).
        k: experts per token.
        expert_outputs: shape (n, E, d_out). Entry [t, e] is expert e's
            output for token t. Treat it as fixed.
        grad_output: shape (n, d_out), the loss gradient with respect to each
            token's mixed output. Treat it as fixed.

    Returns:
        The sum over tokens and chosen experts of
        gate_weight * (grad_output[t] @ expert_outputs[t, expert]).
    """
    topk_indices, gate_weights, _ = router(x, W_gate, k)
    total = 0.0
    for t in range(x.shape[0]):
        for slot, expert in enumerate(topk_indices[t]):
            total += gate_weights[t, slot] * (grad_output[t] @ expert_outputs[t, expert])
    return float(total)


def router_backward(
    x: np.ndarray,
    W_gate: np.ndarray,
    k: int,
    expert_outputs: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Gradient of surrogate_loss with respect to W_gate, shape (d, E).

    Build dz of shape (n, E), zero everywhere except at each token's chosen
    experts, where it holds gate_logit_grad. Return x.T @ dz.
    """
    topk_indices, gate_weights, full_probs = router(x, W_gate, k)
    dz = np.zeros((x.shape[0], W_gate.shape[1]))
    for t in range(x.shape[0]):
        chosen = topk_indices[t]
        c = expert_outputs[t, chosen] @ grad_output[t]
        dz[t, chosen] = gate_logit_grad(gate_weights[t], c)
    return x.T @ dz


if __name__ == "__main__":
    g = np.array([0.7311, 0.2689])
    c = np.array([1.0, 3.0])
    print("gradients on the two logits (expect -0.3932, 0.3932):", np.round(gate_logit_grad(g, c), 4))

    rng = np.random.default_rng(0)
    n, d, E, k, d_out = 5, 4, 4, 2, 3
    x = rng.normal(size=(n, d))
    W = rng.normal(size=(d, E))
    outs = rng.normal(size=(n, E, d_out))
    go = rng.normal(size=(n, d_out))
    grad = router_backward(x, W, k, outs, go)

    eps = 1e-6
    W_plus = W.copy()
    W_plus[0, 0] += eps
    W_minus = W.copy()
    W_minus[0, 0] -= eps
    numeric = (surrogate_loss(x, W_plus, k, outs, go) - surrogate_loss(x, W_minus, k, outs, go)) / (2 * eps)
    print(f"entry [0, 0]: analytic {grad[0, 0]:.6f}, finite difference {numeric:.6f}")
