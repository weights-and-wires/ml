"""Reference solution: evaluating language models."""

from __future__ import annotations

import numpy as np


def bits_per_byte(loss_nats_sum: float, n_bytes: int) -> float:
    """Convert a total loss in nats over a text into bits per byte.

    Args:
        loss_nats_sum: The loss summed over every token of the text, in nats.
        n_bytes: The number of bytes of the text.

    Returns:
        The total loss in bits divided by the byte count.
    """
    return float(loss_nats_sum / (n_bytes * np.log(2.0)))


def choose_answer(logprobs: np.ndarray, lengths: np.ndarray, normalise: bool) -> int:
    """Pick the best option of a multiple-choice question.

    Args:
        logprobs: Shape (n_options,). The total log probability of each option.
        lengths: Shape (n_options,). The length of each option, in bytes.
        normalise: When true, divide each total by its length first.

    Returns:
        The index of the highest score. Ties go to the first option.
    """
    logprobs = np.asarray(logprobs, dtype=float)
    scores = logprobs / np.asarray(lengths, dtype=float) if normalise else logprobs
    return int(np.argmax(scores))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct.

    Computes 1 - C(n - c, k) / C(n, k) as 1 - prod_{i=n-c+1}^{n} (1 - k / i),
    so nothing overflows.

    Args:
        n: Samples drawn for the problem.
        c: Samples that passed, 0 <= c <= n.
        k: Attempts allowed, 1 <= k <= n.

    Returns:
        A float in [0, 1].
    """
    i = np.arange(n - c + 1, n + 1, dtype=float)
    return float(1.0 - np.prod(1.0 - k / i))


def bootstrap_ci(scores: np.ndarray, B: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-item scores.

    Args:
        scores: Shape (n_items,). One score per item.
        B: Number of resamples.
        rng: A seeded numpy Generator.
        alpha: One minus the coverage.

    Returns:
        (low, high), the alpha/2 and 1 - alpha/2 percentiles of the resampled means.
    """
    scores = np.asarray(scores, dtype=float)
    idx = rng.integers(0, len(scores), size=(B, len(scores)))
    means = scores[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    scores_a: np.ndarray, scores_b: np.ndarray, B: int, rng: np.random.Generator, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap for mean(scores_b) - mean(scores_a) on the same items.

    Args:
        scores_a: Shape (n_items,). Model A's score on each item.
        scores_b: Shape (n_items,). Model B's score on the same items.
        B: Number of resamples.
        rng: A seeded numpy Generator.
        alpha: One minus the coverage.

    Returns:
        (observed difference, low, high).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def _grams(doc: list, n: int) -> set:
    return {tuple(doc[i : i + n]) for i in range(len(doc) - n + 1)}


def ngram_overlap(train_docs: list, test_doc: list, n: int) -> float:
    """Fraction of the test document's n-grams found in any training document.

    Args:
        train_docs: A list of documents, each a list of tokens.
        test_doc: One document, a list of tokens.
        n: The n-gram length.

    Returns:
        A float in [0, 1]. 0.0 when the test document has fewer than n tokens.
    """
    test = _grams(test_doc, n)
    if not test:
        return 0.0
    seen = set()
    for doc in train_docs:
        seen |= _grams(doc, n)
    return len(test & seen) / len(test)


if __name__ == "__main__":
    print("bpb:", bits_per_byte(40 * np.log(2.0), 40))
    print("raw pick:", choose_answer(np.array([-6.0, -9.0, -12.0]), np.array([3, 6, 12]), False))
    print("normalised pick:", choose_answer(np.array([-6.0, -9.0, -12.0]), np.array([3, 6, 12]), True))
    print("pass@2:", pass_at_k(5, 2, 2))
    rng = np.random.default_rng(0)
    print("ci:", bootstrap_ci((rng.uniform(size=100) < 0.6).astype(float), 500, rng))
    print("overlap:", ngram_overlap([list("xbcdy")], list("abcde"), 3))
