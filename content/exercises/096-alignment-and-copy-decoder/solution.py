"""Reference solution: the alignment matrix and a hand-built copy decoder.

Shapes throughout: a source of `T_x` tokens gives `keys` of shape `(T_x, d_h)`
and `values` of shape `(T_x, d_v)`. A decoder of `T_y` steps gives `queries`
of shape `(T_y, d_s)`, one row per step.
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
    source = [3, 0, 4, 1, 2]
    decoded, alignment = decode_with_attention(source, vocab_size=5)
    print(f"source:  {source}")
    print(f"decoded: {decoded}")
    print(f"alignment diagonal: {np.diag(alignment).round(5)}")

    _, flipped = decode_with_attention(source, vocab_size=5, v_sign=1.0)
    print(f"with v flipped, diagonal: {np.diag(flipped).round(5)}")
    print(f"with v flipped, row 2:    {flipped[2].round(3)}")

    one = mean_summary([0, 1, 2], 3)
    two = mean_summary([2, 1, 0], 3)
    print(f"order-blind summaries: {one.round(3)} and {two.round(3)}")
    print(f"decoded [0, 1, 2]: {decode_with_attention([0, 1, 2], 3)[0]}")
    print(f"decoded [2, 1, 0]: {decode_with_attention([2, 1, 0], 3)[0]}")
