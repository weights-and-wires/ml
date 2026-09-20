"""Checks for the seq2seq attention exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def additive_case(seed: int = 0, Tx: int = 5, d_s: int = 4, d_h: int = 4, d_v: int = 3, d: int = 6):
    rng = np.random.default_rng(seed)
    query = rng.normal(size=d_s)
    keys = rng.normal(size=(Tx, d_h))
    values = rng.normal(size=(Tx, d_v))
    W1 = 0.4 * rng.normal(size=(d, d_s))
    W2 = 0.4 * rng.normal(size=(d, d_h))
    v = 0.4 * rng.normal(size=d)
    return query, keys, values, W1, W2, v


@check("additive_attention returns a valid probability distribution over source positions")
def test_additive_weights_valid(sub):
    query, keys, values, W1, W2, v = additive_case()
    fn = has(sub, "additive_attention")
    context, weights = fn(query, keys, values, W1, W2, v)

    shape(weights, (keys.shape[0],), label="additive weights")
    shape(context, (values.shape[1],), label="additive context")
    if np.any(weights < -1e-10):
        fail(f"attention weights must be non-negative, got minimum {weights.min():.3g}")
    total = float(weights.sum())
    if abs(total - 1.0) > 1e-8:
        fail(f"attention weights must sum to one, got {total:.6f}")


@check("multiplicative_attention returns a valid probability distribution, dot and general forms")
def test_multiplicative_weights_valid(sub):
    query, keys, values, _, _, _ = additive_case(seed=1)
    fn = has(sub, "multiplicative_attention")

    _, weights_dot = fn(query, keys, values)
    total_dot = float(weights_dot.sum())
    if abs(total_dot - 1.0) > 1e-8 or np.any(weights_dot < -1e-10):
        fail(f"dot-product weights are not a valid distribution: sum {total_dot:.6f}, "
             f"min {weights_dot.min():.3g}")

    rng = np.random.default_rng(2)
    W = 0.4 * rng.normal(size=(query.shape[0], keys.shape[1]))
    _, weights_general = fn(query, keys, values, W)
    total_general = float(weights_general.sum())
    if abs(total_general - 1.0) > 1e-8 or np.any(weights_general < -1e-10):
        fail(f"general bilinear weights are not a valid distribution: sum {total_general:.6f}, "
             f"min {weights_general.min():.3g}")


@check("the context vector is a convex combination of the values")
def test_context_is_convex_combination(sub):
    query, keys, values, W1, W2, v = additive_case(seed=3)
    additive = has(sub, "additive_attention")
    multiplicative = has(sub, "multiplicative_attention")

    for context, weights in (
        additive(query, keys, values, W1, W2, v),
        multiplicative(query, keys, values),
    ):
        want = weights @ values
        close(context, want, atol=1e-8, label="context as weights @ values")
        lo, hi = values.min(axis=0), values.max(axis=0)
        if np.any(context < lo - 1e-6) or np.any(context > hi + 1e-6):
            fail(
                "a convex combination of the values cannot leave their "
                f"per-coordinate range [{lo}, {hi}], but the context is {context}"
            )


@check("attention_backward matches finite differences on every argument")
def test_attention_backward(sub):
    query, keys, values, W1, W2, v = additive_case(seed=4)
    forward = has(sub, "additive_attention")
    backward = has(sub, "attention_backward")

    rng = np.random.default_rng(40)
    dcontext = 0.3 * rng.normal(size=values.shape[1])

    grads = backward(dcontext, query, keys, values, W1, W2, v)
    names = ["query", "keys", "values", "W1", "W2", "v"]
    args = {"query": query, "keys": keys, "values": values, "W1": W1, "W2": W2, "v": v}

    for name, grad in zip(names, grads):
        def loss(arr, name=name):
            local = dict(args)
            local[name] = arr
            context, _ = forward(local["query"], local["keys"], local["values"],
                                  local["W1"], local["W2"], local["v"])
            return float(np.dot(context, dcontext))

        grad_fd = finite_difference(loss, args[name])
        close(grad, grad_fd, atol=1e-4, rtol=1e-3, label=f"gradient wrt {name}")


@check("additive attention reduces to a dot product when its parameters are set to emulate one")
def test_additive_emulates_dot_product(sub):
    additive = has(sub, "additive_attention")
    multiplicative = has(sub, "multiplicative_attention")

    rng = np.random.default_rng(5)
    d = 4
    Tx = 6
    query = rng.normal(size=d)
    keys = rng.normal(size=(Tx, d))
    values = rng.normal(size=(Tx, 3))

    c = 1e-4
    W1 = c * np.eye(d)
    W2 = c * np.eye(d)
    v = query  # v = q makes the constant ||q||^2 offset cancel in the softmax.

    _, weights_additive = additive(query, keys, values, W1, W2, v)
    _, weights_dot = multiplicative(c * query, keys, values)

    close(
        weights_additive, weights_dot, atol=1e-6, rtol=1e-5,
        label="additive weights against a scaled dot product",
    )


@check("decode_with_attention exactly reproduces the source tokens on a copy task")
def test_decode_copies_source(sub):
    fn = has(sub, "decode_with_attention")
    source = [3, 0, 4, 1, 2]
    decoded, alignment = fn(source, vocab_size=5)

    if list(decoded) != source:
        fail(f"the copy task should reproduce the source exactly: got {decoded}, wanted {source}")
    shape(alignment, (len(source), len(source)), label="alignment matrix")


@check("decode_with_attention's alignment matrix is close to the identity")
def test_decode_alignment_is_identity(sub):
    fn = has(sub, "decode_with_attention")
    source = [4, 2, 0, 3, 1]
    _, alignment = fn(source, vocab_size=5)

    identity = np.eye(len(source))
    close(
        alignment, identity, atol=1e-2, rtol=0.0,
        label="alignment matrix against the identity",
    )
