"""Reference solution: backpropagation through additive attention, the alignment matrix, and a copy decoder.

Shapes throughout: `query` is `(d_s,)`, `keys` is `(T, d_h)`, `values` is
`(T, d_v)`, `W1` is `(h, d_s)`, `W2` is `(h, d_h)`, and `v` is `(h,)`.
A source of `T_x` tokens gives `keys` of shape `(T_x, d_h)`; a decoder of
`T_y` steps gives `queries` of shape `(T_y, d_s)`, one row per step.

A name that starts with `d`, such as `dweights`, holds the slope of the loss
with respect to the thing that follows. The loss is `dcontext @ context`, for
a fixed vector `dcontext`, so the slope with respect to the context is exactly
`dcontext`.
"""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    exp = np.exp(z - z.max())
    return exp / exp.sum()


def attend(scores: np.ndarray, values: np.ndarray):
    weights = softmax(scores)  # (T,)
    context = weights @ values  # (d_v,)
    return context, weights


def additive_scores(query, keys, W1, W2, v):
    u = W1 @ query + keys @ W2.T  # (T, h)
    return np.tanh(u) @ v  # (T,)


def additive_attention(query, keys, values, W1, W2, v):
    scores = additive_scores(query, keys, W1, W2, v)
    return attend(scores, values)


def softmax_backward(weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
    """Send the slope on the softmax weights back to the scores.

    Args:
        weights: The softmax output, shape (T,).
        dweights: The slope of the loss with respect to the weights, shape (T,).

    Returns:
        dscores, shape (T,): the slope of the loss with respect to the scores.
    """
    return weights * (dweights - np.sum(weights * dweights))  # (T,)


def context_backward(dcontext: np.ndarray, weights: np.ndarray, values: np.ndarray):
    """Send the slope on the context back through `context = weights @ values`.

    Args:
        dcontext: Slope with respect to the context, shape (d_v,).
        weights: Shape (T,).
        values: Shape (T, d_v).

    Returns:
        A tuple `(dweights, dvalues)` with shapes (T,) and (T, d_v).
    """
    dweights = values @ dcontext  # (T,)
    dvalues = np.outer(weights, dcontext)  # (T, d_v)
    return dweights, dvalues


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
    # Forward pass again, keeping what the backward pass needs.
    u = W1 @ query + keys @ W2.T  # (T, h)
    t = np.tanh(u)  # (T, h)
    scores = t @ v  # (T,)
    weights = softmax(scores)  # (T,)

    dweights, dvalues = context_backward(dcontext, weights, values)
    dscores = softmax_backward(weights, dweights)  # (T,)

    dv = t.T @ dscores  # (h,)
    dt = np.outer(dscores, v)  # (T, h)
    du = dt * (1.0 - t**2)  # (T, h)

    du_sum = du.sum(axis=0)  # (h,): the query was copied into every row
    dW1 = np.outer(du_sum, query)  # (h, d_s)
    dquery = W1.T @ du_sum  # (d_s,)
    dW2 = du.T @ keys  # (h, d_h)
    dkeys = du @ W2  # (T, d_h)
    return dquery, dkeys, dvalues, dW1, dW2, dv


def alignment_matrix(queries, keys, values, W1, W2, v):
    """Run additive attention once per decoder step and stack the results.

    Args:
        queries: One query per decoder step, shape (T_y, d_s).
        keys: Shape (T_x, d_h).
        values: Shape (T_x, d_v).
        W1, W2, v: The scorer's parameters, as in `additive_attention`.

    Returns:
        A tuple `(contexts, A)`.
        contexts: One context per step, shape (T_y, d_v).
        A: The alignment matrix, shape (T_y, T_x). Row t holds the attention
            weights of step t, so each row sums to one.
    """
    contexts = []
    rows = []
    for query in queries:
        context, weights = additive_attention(query, keys, values, W1, W2, v)
        contexts.append(context)
        rows.append(weights)
    return np.stack(contexts), np.stack(rows)


def mean_summary(source_tokens, vocab_size: int) -> np.ndarray:
    """The order-blind fixed-size summary: the mean of the tokens' one-hot rows.

    Args:
        source_tokens: A sequence of integers in `range(vocab_size)`.
        vocab_size: Size of the vocabulary.

    Returns:
        A vector of shape (vocab_size,) that sums to one.
    """
    one_hot = np.eye(vocab_size)[np.asarray(source_tokens)]  # (T, vocab_size)
    return one_hot.mean(axis=0)


def decode_with_attention(source_tokens, vocab_size: int, v_sign: float = -1.0):
    """Copy a source with a hand-built additive-attention head.

    Build fixed (not learned) parameters so that decoder step t attends almost
    entirely to source position t, then read off the token stored there.

    Use `T = len(source_tokens)`. The queries and the keys are both
    `10 * np.eye(T)`, so step t and position t share the same direction. The
    values are the one-hot rows of the source tokens. `W1` and `W2` are the
    identity, and `v = v_sign * 10 * np.ones(T)`.

    Args:
        source_tokens: A sequence of `T` integers in `range(vocab_size)`.
        vocab_size: Size of the vocabulary.
        v_sign: -1.0 for the copying decoder, +1.0 to flip the sign of v.

    Returns:
        A tuple `(decoded, alignment)`.
        decoded: A list of `T` ints, the argmax of each step's context.
        alignment: Shape (T, T), the weights of every step.
    """
    T = len(source_tokens)
    positions = 10.0 * np.eye(T)  # queries and keys: step t and position t share a direction
    values = np.eye(vocab_size)[np.asarray(source_tokens)]  # (T, vocab_size)
    identity = np.eye(T)
    v = v_sign * 10.0 * np.ones(T)

    contexts, alignment = alignment_matrix(positions, positions, values, identity, identity, v)
    decoded = [int(np.argmax(row)) for row in contexts]
    return decoded, alignment


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
    c_hi, _ = additive_attention(query + np.array([eps, 0.0]), keys, values, eye, eye, v)
    c_lo, _ = additive_attention(query - np.array([eps, 0.0]), keys, values, eye, eye, v)
    print(f"finite difference for dquery[0]: {(c_hi - c_lo) @ dcontext / (2 * eps):.4f}")

    source = [3, 0, 4, 1, 2]
    decoded, alignment = decode_with_attention(source, vocab_size=5)
    print(f"source:  {source}")
    print(f"decoded: {decoded}")
    print(f"alignment diagonal: {np.diag(alignment).round(5)}")
    _, flipped = decode_with_attention(source, vocab_size=5, v_sign=1.0)
    print(f"with v flipped, diagonal: {np.diag(flipped).round(5)}")

    one = mean_summary([0, 1, 2], 3)
    two = mean_summary([2, 1, 0], 3)
    print(f"order-blind summaries: {one.round(3)} and {two.round(3)}")
    print(f"decoded [0, 1, 2]: {decode_with_attention([0, 1, 2], 3)[0]}")
    print(f"decoded [2, 1, 0]: {decode_with_attention([2, 1, 0], 3)[0]}")
