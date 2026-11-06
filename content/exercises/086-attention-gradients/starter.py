"""Backpropagation through additive attention.

Shapes throughout: `query` is `(d_s,)`, `keys` is `(T, d_h)`, `values` is
`(T, d_v)`, `W1` is `(h, d_s)`, `W2` is `(h, d_h)`, and `v` is `(h,)`.

A name that starts with `d`, such as `dweights`, holds the slope of the loss
with respect to the thing that follows. In this chapter the loss is
`dcontext @ context`, for a fixed vector `dcontext`, so the slope of the loss
with respect to the context is exactly `dcontext`.
"""

import numpy as np


# --- Written in the chapter "Why one vector is not enough: attention as a soft lookup" ---


def softmax(z: np.ndarray) -> np.ndarray:
    exp = np.exp(z - z.max())
    return exp / exp.sum()


def attend(scores: np.ndarray, values: np.ndarray):
    weights = softmax(scores)  # (T,)
    context = weights @ values  # (d_v,)
    return context, weights


# --- Written in the chapter "Additive attention: let a small network score the match" ---


def additive_scores(query, keys, W1, W2, v):
    u = W1 @ query + keys @ W2.T  # (T, h)
    return np.tanh(u) @ v  # (T,)


def additive_attention(query, keys, values, W1, W2, v):
    scores = additive_scores(query, keys, W1, W2, v)
    return attend(scores, values)


# --- Written in this chapter ---


def softmax_backward(weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
    """Send the slope on the softmax weights back to the scores.

    Args:
        weights: The softmax output, shape (T,).
        dweights: The slope of the loss with respect to the weights, shape (T,).

    Returns:
        dscores, shape (T,): the slope of the loss with respect to the scores.
    """
    raise NotImplementedError("softmax_backward")


def context_backward(dcontext: np.ndarray, weights: np.ndarray, values: np.ndarray):
    """Send the slope on the context back through `context = weights @ values`.

    Args:
        dcontext: Slope with respect to the context, shape (d_v,).
        weights: Shape (T,).
        values: Shape (T, d_v).

    Returns:
        A tuple `(dweights, dvalues)` with shapes (T,) and (T, d_v).
    """
    raise NotImplementedError("context_backward")


def attention_backward(
    dcontext: np.ndarray,
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
):
    """Backpropagate through `additive_attention`.

    Rerun the forward pass inside this function, then walk it backward: the
    blend, the softmax, the dot with v, the tanh, and the two projections.
    Use `softmax_backward` and `context_backward` for the first two steps.

    Args:
        dcontext: Slope with respect to the context, shape (d_v,).
        query, keys, values, W1, W2, v: The arguments of `additive_attention`.

    Returns:
        A tuple `(dquery, dkeys, dvalues, dW1, dW2, dv)`, each with the shape
        of the argument it belongs to.
    """
    raise NotImplementedError("attention_backward")


if __name__ == "__main__":
    query = np.array([1.0, 0.0])
    keys = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = np.array([[10.0, 0.0], [0.0, 10.0]])
    eye = np.eye(2)
    v = np.array([1.0, 1.0])
    dcontext = np.array([1.0, 0.0])

    _, weights = additive_attention(query, keys, values, eye, eye, v)
    dweights, dvalues = context_backward(dcontext, weights, values)
    dscores = softmax_backward(weights, dweights)
    print(f"dweights {dweights.round(3)}, dscores {dscores.round(3)}, sum {dscores.sum():.1e}")

    dquery, dkeys, dvalues, dW1, dW2, dv = attention_backward(dcontext, query, keys, values, eye, eye, v)
    print(f"dquery {dquery.round(3)}")
    print(f"dv {dv.round(3)}")

    # Check one number against a nudge of 1e-6.
    eps = 1e-6
    nudged = query + np.array([eps, 0.0])
    c_hi, _ = additive_attention(nudged, keys, values, eye, eye, v)
    c_lo, _ = additive_attention(query - np.array([eps, 0.0]), keys, values, eye, eye, v)
    print(f"finite difference for dquery[0]: {(c_hi - c_lo) @ dcontext / (2 * eps):.4f}")
