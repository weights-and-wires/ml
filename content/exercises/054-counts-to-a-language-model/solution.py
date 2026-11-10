"""Reference solution: counts, a window language model, and skip-gram."""

from __future__ import annotations

import numpy as np


def bigram_counts(prev: np.ndarray, nxt: np.ndarray, vocab_size: int) -> np.ndarray:
    """Count how often each word follows each other word.

    Args:
        prev: Integer ids of the previous word, shape (N,).
        nxt: Integer ids of the next word, shape (N,).
        vocab_size: Number of rows and columns, V.

    Returns:
        Float array of shape (V, V). Entry [i, j] is the number of times
        nxt is j when prev is i. A repeated pair adds.
    """
    counts = np.zeros((vocab_size, vocab_size), dtype=float)
    np.add.at(counts, (np.asarray(prev, dtype=int), np.asarray(nxt, dtype=int)), 1.0)
    return counts


def add_k_probs(counts: np.ndarray, k: float) -> np.ndarray:
    """Turn counts into probabilities by adding k to every cell.

    Args:
        counts: Array of shape (V, V).
        k: Positive count added to every cell, including zeros.

    Returns:
        Float array of shape (V, V). Each row is positive and sums to 1.
        An all-zero row becomes the uniform distribution 1/V.
    """
    smoothed = np.asarray(counts, dtype=float) + k
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def mean_nll(probs: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> float:
    """Mean of -ln p over the pairs.

    Args:
        probs: Probabilities of shape (V, V).
        prev: Previous-word ids, shape (N,).
        nxt: Next-word ids, shape (N,).

    Returns:
        The mean natural-log cost, a Python float.
    """
    chosen = np.asarray(probs, dtype=float)[np.asarray(prev, dtype=int), np.asarray(nxt, dtype=int)]
    return float(-np.mean(np.log(chosen)))


def perplexity(nll: float) -> float:
    """Read a mean negative log likelihood as a count of choices.

    Args:
        nll: Mean of -ln p.

    Returns:
        e raised to nll. A uniform model over V words returns V.
    """
    return float(np.exp(nll))


def row_softmax(logits: np.ndarray) -> np.ndarray:
    """Turn each row of scores into probabilities.

    Args:
        logits: Scores of shape (N, V).

    Returns:
        Float array of the same shape. Each row is positive and sums to 1.
        A row of equal scores, however large, is uniform.
    """
    scores = np.asarray(logits, dtype=float)
    shifted = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def embedding_forward(table: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Copy one row of the table per index.

    Args:
        table: Embedding table of shape (V, D).
        indices: Integer ids of shape (N,).

    Returns:
        Float array of shape (N, D). A repeated index repeats the row.
    """
    return np.asarray(table, dtype=float)[np.asarray(indices, dtype=int)]


def embedding_backward(
    grad_out: np.ndarray, indices: np.ndarray, vocab_size: int
) -> np.ndarray:
    """Scatter-add the lookup gradient back into the table.

    Args:
        grad_out: Gradient on the looked-up rows, shape (N, D).
        indices: Integer ids of shape (N,), the rows that were read.
        vocab_size: Number of rows in the table.

    Returns:
        Float array of shape (vocab_size, D). Rows that were not read
        stay 0. A repeated index adds the two gradients.
    """
    outgoing = np.asarray(grad_out, dtype=float)
    gradient = np.zeros((vocab_size, outgoing.shape[1]), dtype=float)
    np.add.at(gradient, np.asarray(indices, dtype=int), outgoing)
    return gradient


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def skipgram_loss_and_grad(
    center: np.ndarray, positive: np.ndarray, negatives: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Negative-sampling loss for one center word.

    Args:
        center: Center vector, shape (D,).
        positive: Context vector of the real neighbour, shape (D,).
        negatives: Context vectors of K noise words, shape (K, D).

    Returns:
        The loss -ln σ(c·p) - sum ln σ(-c·n), as a Python float;
        the gradient on center, shape (D,);
        the gradient on the real context vector, shape (D,);
        the gradient on the noise vectors, shape (K, D).
    """
    center_vec = np.asarray(center, dtype=float)
    positive_vec = np.asarray(positive, dtype=float)
    negative_vecs = np.asarray(negatives, dtype=float)
    positive_score = float(center_vec @ positive_vec)
    negative_scores = negative_vecs @ center_vec
    loss = float(_softplus(-positive_score) + np.sum(_softplus(negative_scores)))
    positive_gate = float(_sigmoid(np.asarray(positive_score)) - 1.0)
    negative_gates = _sigmoid(negative_scores)
    grad_center = positive_gate * positive_vec + (negative_gates[:, None] * negative_vecs).sum(axis=0)
    grad_positive = positive_gate * center_vec
    grad_negatives = negative_gates[:, None] * center_vec
    return loss, grad_center, grad_positive, grad_negatives


def _window_batch(
    sentences: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    for sentence in sentences:
        for index in range(window, len(sentence)):
            contexts.append(sentence[index - window : index])
            targets.append(int(sentence[index]))
    return np.asarray(contexts, dtype=int), np.asarray(targets, dtype=int)


def train_window_lm(
    sentences: np.ndarray,
    vocab_size: int,
    window: int,
    dim: int,
    hidden: int,
    steps: int,
    lr: float,
    seed: int,
    query_context: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Train a tanh window language model and read one context.

    Draw the embedding, then the first weight, then the second, from
    ``np.random.default_rng(seed)`` as a normal of scale 0.1. Biases start
    at 0. Each step is full-batch gradient descent on the mean negative
    log likelihood of every window in ``sentences``.

    Args:
        sentences: Integer token ids, shape (S, T).
        vocab_size: V.
        window: Number of previous tokens in the context.
        dim: Embedding width D.
        hidden: Hidden width H.
        steps: Number of full-batch updates.
        lr: Learning rate.
        seed: Seed for the initial draw.
        query_context: Integer ids, shape (window,).

    Returns:
        The mean negative log likelihood after the updates, and the
        probability of every next word given ``query_context``, shape (V,).
    """
    rng = np.random.default_rng(seed)
    table = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    weight_1 = rng.normal(0.0, 0.1, size=(window * dim, hidden))
    bias_1 = np.zeros(hidden)
    weight_2 = rng.normal(0.0, 0.1, size=(hidden, vocab_size))
    bias_2 = np.zeros(vocab_size)
    context_ids, target_ids = _window_batch(sentences, window)
    count = len(target_ids)
    for _ in range(steps):
        flat = table[context_ids].reshape(count, -1)
        hidden_act = np.tanh(flat @ weight_1 + bias_1)
        logits = hidden_act @ weight_2 + bias_2
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        grad_logits = probs.copy()
        grad_logits[np.arange(count), target_ids] -= 1.0
        grad_logits /= count
        grad_hidden = grad_logits @ weight_2.T * (1.0 - hidden_act**2)
        grad_flat = (grad_hidden @ weight_1.T).reshape(count, window, dim)
        grad_table = np.zeros_like(table)
        np.add.at(grad_table, context_ids, grad_flat)
        table -= lr * grad_table
        weight_1 -= lr * (flat.T @ grad_hidden)
        bias_1 -= lr * grad_hidden.sum(axis=0)
        weight_2 -= lr * (hidden_act.T @ grad_logits)
        bias_2 -= lr * grad_logits.sum(axis=0)
    flat = table[context_ids].reshape(count, -1)
    hidden_act = np.tanh(flat @ weight_1 + bias_1)
    logits = hidden_act @ weight_2 + bias_2
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    nll = float(-np.mean(np.log(probs[np.arange(count), target_ids])))
    query = np.asarray(query_context, dtype=int)
    query_flat = table[query].reshape(1, -1)
    query_hidden = np.tanh(query_flat @ weight_1 + bias_1)
    query_logits = (query_hidden @ weight_2 + bias_2).ravel()
    query_logits = query_logits - query_logits.max()
    dist = np.exp(query_logits)
    dist /= dist.sum()
    return nll, dist


def _pairs(sentences: np.ndarray, window: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for sentence in sentences:
        length = len(sentence)
        for center_at in range(length):
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                context_at = center_at + offset
                if 0 <= context_at < length:
                    found.append((int(sentence[center_at]), int(sentence[context_at])))
    return found


def _noise(sentences: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.bincount(np.asarray(sentences, dtype=int).ravel(), minlength=vocab_size).astype(float)
    weights = np.power(counts, 0.75)
    return weights / weights.sum()


def train_skipgram(
    sentences: np.ndarray,
    vocab_size: int,
    window: int,
    dim: int,
    epochs: int,
    lr: float,
    n_negative: int,
    seed: int,
) -> np.ndarray:
    """Train skip-gram with negative sampling and return the center table.

    Draw the center table from ``np.random.default_rng(seed)`` as a normal
    of scale 0.1, then the context table the same way. Pairs stay inside
    each sentence, both directions, skipping offset 0. Each epoch shuffles
    with that generator. Noise is the unigram raised to 0.75, and a draw
    equal to the real context word is rejected.

    Args:
        sentences: Integer token ids, shape (S, T).
        vocab_size: V.
        window: How many tokens either side of the center to pair.
        dim: Vector width D.
        epochs: Passes over the pair list.
        lr: Learning rate.
        n_negative: Noise words per real pair.
        seed: Seed for the initial draw and the shuffle.

    Returns:
        Center table, shape (V, D).
    """
    rng = np.random.default_rng(seed)
    center = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    context = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    pair_list = _pairs(sentences, window)
    noise = _noise(sentences, vocab_size)
    for _ in range(epochs):
        order = rng.permutation(len(pair_list))
        for index in order:
            center_id, positive_id = pair_list[int(index)]
            negatives: list[int] = []
            while len(negatives) < n_negative:
                draw = int(rng.choice(vocab_size, p=noise))
                if draw != positive_id:
                    negatives.append(draw)
            negative_ids = np.asarray(negatives, dtype=int)
            center_vec = center[center_id].copy()
            positive_vec = context[positive_id].copy()
            negative_vecs = context[negative_ids].copy()
            loss, grad_center, grad_positive, grad_negatives = skipgram_loss_and_grad(
                center_vec, positive_vec, negative_vecs
            )
            del loss
            center[center_id] = center_vec - lr * grad_center
            context[positive_id] = positive_vec - lr * grad_positive
            for slot, negative_id in enumerate(negative_ids):
                context[negative_id] = context[negative_id] - lr * grad_negatives[slot]
    return center


def notebook(repeats: int = 1) -> np.ndarray:
    """Four lines, each `start` plus three words, repeated `repeats` times.

    Indices are start, big, small, dog, hound, cat, kitten, runs, sits.
    """
    lines = np.array(
        [
            [0, 1, 3, 7],
            [0, 1, 4, 7],
            [0, 2, 5, 8],
            [0, 2, 6, 8],
        ],
        dtype=int,
    )
    return np.repeat(lines, repeats, axis=0)


if __name__ == "__main__":
    prev = np.array([0, 1, 2, 0, 1, 3])
    nxt = np.array([1, 2, 4, 1, 3, 4])
    counts = bigram_counts(prev, nxt, 5)
    print("counts")
    print(counts.astype(int))
    probs = add_k_probs(counts, 1.0)
    nll = mean_nll(probs, prev, nxt)
    print(f"mean nll {nll:.4f}  perplexity {perplexity(nll):.4f}")
    lines = notebook(4)
    loss, dist = train_window_lm(
        lines, 9, window=2, dim=8, hidden=16, steps=80, lr=0.5, seed=0,
        query_context=np.array([1, 3]),
    )
    print(f"window nll {loss:.3f}  P(runs) {dist[7]:.3f}  P(sits) {dist[8]:.3f}")
    center = train_skipgram(
        lines, 9, window=1, dim=8, epochs=20, lr=0.4, n_negative=4, seed=0,
    )

    def cosine(i: int, j: int) -> float:
        a, b = center[i], center[j]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"cosine dog-hound {cosine(3, 4):.3f}  dog-sits {cosine(3, 8):.3f}")
