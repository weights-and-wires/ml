"""From counts to a next-word model, and from a word to a vector.

Fill in the ten functions below. The tally uses a five-word vocabulary.
The training notebook uses nine words, listed in `notebook`.
"""

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
    raise NotImplementedError("write bigram_counts")


def add_k_probs(counts: np.ndarray, k: float) -> np.ndarray:
    """Turn counts into probabilities by adding k to every cell.

    Args:
        counts: Array of shape (V, V).
        k: Positive count added to every cell, including zeros.

    Returns:
        Float array of shape (V, V). Each row is positive and sums to 1.
        An all-zero row becomes the uniform distribution 1/V.
    """
    raise NotImplementedError("write add_k_probs")


def mean_nll(probs: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> float:
    """Mean of -ln p over the pairs.

    Args:
        probs: Probabilities of shape (V, V).
        prev: Previous-word ids, shape (N,).
        nxt: Next-word ids, shape (N,).

    Returns:
        The mean natural-log cost, a Python float.
    """
    raise NotImplementedError("write mean_nll")


def perplexity(nll: float) -> float:
    """Read a mean negative log likelihood as a count of choices.

    Args:
        nll: Mean of -ln p.

    Returns:
        e raised to nll. A uniform model over V words returns V.
    """
    raise NotImplementedError("write perplexity")


def row_softmax(logits: np.ndarray) -> np.ndarray:
    """Turn each row of scores into probabilities.

    Args:
        logits: Scores of shape (N, V).

    Returns:
        Float array of the same shape. Each row is positive and sums to 1.
        A row of equal scores, however large, is uniform.
    """
    raise NotImplementedError("write row_softmax")


def embedding_forward(table: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Copy one row of the table per index.

    Args:
        table: Embedding table of shape (V, D).
        indices: Integer ids of shape (N,).

    Returns:
        Float array of shape (N, D). A repeated index repeats the row.
    """
    raise NotImplementedError("write embedding_forward")


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
    raise NotImplementedError("write embedding_backward")


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
    raise NotImplementedError("write skipgram_loss_and_grad")


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
    raise NotImplementedError("write train_window_lm")


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
    raise NotImplementedError("write train_skipgram")


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
