"""Additive and multiplicative attention.

You implement both attention scoring functions, the backward pass for the
additive form, and a hand-built decode loop that demonstrates a correctly
aligned attention matrix on a copy task.

Shapes throughout: `query` is `(d_s,)`, `keys` is `(Tx, d_h)`, `values` is
`(Tx, d_v)`. A single query attends over all `Tx` source positions at once.
"""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """A numerically stable softmax over the last axis of a 1-D array.

    Args:
        z: Scores, shape (Tx,).

    Returns:
        A probability vector of shape (Tx,), non-negative and summing to one.
    """
    shifted = z - z.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def additive_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
):
    """Bahdanau additive attention for a single query.

    Computes `scores[i] = v @ tanh(W1 @ query + W2 @ keys[i])`, softmaxes the
    scores across the `Tx` source positions, and returns the resulting
    weighted average of `values`.

    Args:
        query: Decoder state, shape (d_s,).
        keys: Encoder states, shape (Tx, d_h).
        values: Vectors to combine, shape (Tx, d_v). Typically equal to keys.
        W1: Query projection, shape (d, d_s).
        W2: Key projection, shape (d, d_h).
        v: Scoring vector, shape (d,).

    Returns:
        A tuple `(context, weights)`.
        context: Weighted combination of values, shape (d_v,).
        weights: Attention weights, shape (Tx,), non-negative and summing to
            one.
    """
    raise NotImplementedError("additive_attention")


def multiplicative_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W: np.ndarray | None = None,
):
    """Luong multiplicative attention for a single query.

    With `W` given, computes `scores[i] = query @ W @ keys[i]`. With `W` equal
    to None, computes the plain dot product `scores[i] = query @ keys[i]`,
    which requires `query.shape == (keys.shape[1],)`.

    Args:
        query: Decoder state, shape (d_s,).
        keys: Encoder states, shape (Tx, d_h).
        values: Vectors to combine, shape (Tx, d_v).
        W: Bilinear weight, shape (d_s, d_h), or None for the dot-product form.

    Returns:
        A tuple `(context, weights)`, with the same shapes as
        `additive_attention`.
    """
    raise NotImplementedError("multiplicative_attention")


def attention_backward(
    dcontext: np.ndarray,
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
):
    """Backpropagate through additive_attention.

    Recomputes the forward pass internally, then returns the gradient of
    `dcontext . context` with respect to every argument.

    Args:
        dcontext: Upstream gradient on the context vector, shape (d_v,).
        query, keys, values, W1, W2, v: The same arguments passed to
            `additive_attention`.

    Returns:
        A tuple `(dquery, dkeys, dvalues, dW1, dW2, dv)`, one gradient per
        input argument, each with the same shape as that argument.
    """
    raise NotImplementedError("attention_backward")


def decode_with_attention(source_tokens, vocab_size: int):
    """Greedily decode a copy task with a hand-built additive-attention head.

    Builds fixed (not learned) attention parameters from position vectors so
    that decoder step t attends almost entirely to source position t, then
    reads off the source token stored there.

    Args:
        source_tokens: A sequence of `Tx` distinct integers in
            `range(vocab_size)`.
        vocab_size: Size of the token vocabulary.

    Returns:
        A tuple `(decoded_tokens, alignment)`.
        decoded_tokens: A list of `Tx` integers, the greedily decoded tokens.
        alignment: The attention weights at every step, shape (Tx, Tx), row t
            holding the weights used to produce decoded_tokens[t].
    """
    raise NotImplementedError("decode_with_attention")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Tx, d_s, d_h, d_v, d = 5, 4, 4, 3, 6

    query = rng.normal(size=d_s)
    keys = rng.normal(size=(Tx, d_h))
    values = rng.normal(size=(Tx, d_v))
    W1 = 0.3 * rng.normal(size=(d, d_s))
    W2 = 0.3 * rng.normal(size=(d, d_h))
    v = 0.3 * rng.normal(size=d)

    context, weights = additive_attention(query, keys, values, W1, W2, v)
    print(f"additive context shape {context.shape}, weights sum {weights.sum():.6f}")

    context_mult, weights_mult = multiplicative_attention(query, keys, values)
    print(f"multiplicative (needs d_s == d_h) weights sum {weights_mult.sum():.6f}")

    dcontext = rng.normal(size=d_v)
    grads = attention_backward(dcontext, query, keys, values, W1, W2, v)
    print(f"dquery shape {grads[0].shape}, dW1 shape {grads[3].shape}")

    source = [3, 0, 4, 1, 2]
    decoded, alignment = decode_with_attention(source, vocab_size=5)
    print(f"source:  {source}")
    print(f"decoded: {decoded}")
    print(f"alignment diagonal: {np.diag(alignment).round(4)}")
