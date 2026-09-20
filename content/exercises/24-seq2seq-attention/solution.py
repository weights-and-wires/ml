"""Reference solution for the seq2seq attention exercise."""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
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
    u = (W1 @ query) + keys @ W2.T  # shape (Tx, d)
    scores = np.tanh(u) @ v  # shape (Tx,)
    weights = softmax(scores)
    context = weights @ values
    return context, weights


def multiplicative_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W: np.ndarray | None = None,
):
    if W is None:
        scores = keys @ query
    else:
        scores = keys @ (W.T @ query)
    weights = softmax(scores)
    context = weights @ values
    return context, weights


def attention_backward(
    dcontext: np.ndarray,
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
):
    # Recompute the forward pass.
    u = (W1 @ query) + keys @ W2.T  # (Tx, d)
    t = np.tanh(u)  # (Tx, d)
    scores = t @ v  # (Tx,)
    weights = softmax(scores)  # (Tx,)
    # context = weights @ values, not needed further for the backward pass.

    dvalues = np.outer(weights, dcontext)  # (Tx, d_v)
    dweights = values @ dcontext  # (Tx,)

    # Softmax Jacobian-vector product: d(softmax)/dz applied to dweights.
    dscores = weights * (dweights - np.sum(weights * dweights))  # (Tx,)

    dv = t.T @ dscores  # (d,)
    dt = np.outer(dscores, v)  # (Tx, d)
    du = dt * (1.0 - t ** 2)  # (Tx, d)

    du_sum = du.sum(axis=0)  # (d,)
    dW1 = np.outer(du_sum, query)  # (d, d_s)
    dquery = W1.T @ du_sum  # (d_s,)

    dW2 = du.T @ keys  # (d, d_h)
    dkeys = du @ W2  # (Tx, d_h)

    return dquery, dkeys, dvalues, dW1, dW2, dv


def decode_with_attention(source_tokens, vocab_size: int):
    tx = len(source_tokens)
    scale = 10.0
    bias = 10.0

    positions = scale * np.eye(tx)  # query t and key i are scale * e_t, scale * e_i
    values = np.eye(vocab_size)[np.asarray(source_tokens)]
    identity = np.eye(tx)
    v = -bias * np.ones(tx)

    decoded = []
    alignment = np.zeros((tx, tx))
    for t in range(tx):
        query = positions[t]
        context, weights = additive_attention(query, positions, values, identity, identity, v)
        alignment[t] = weights
        decoded.append(int(np.argmax(context)))
    return decoded, alignment


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
