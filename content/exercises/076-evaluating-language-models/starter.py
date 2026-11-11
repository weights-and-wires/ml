"""Exercise: evaluating language models."""

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
    raise NotImplementedError("write bits_per_byte")


def choose_answer(logprobs: np.ndarray, lengths: np.ndarray, normalise: bool) -> int:
    """Pick the best option of a multiple-choice question.

    Args:
        logprobs: Shape (n_options,). The total log probability of each option.
        lengths: Shape (n_options,). The length of each option, in bytes.
        normalise: When true, divide each total by its length first.

    Returns:
        The index of the highest score. Ties go to the first option.
    """
    raise NotImplementedError("write choose_answer")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct.

    The answer is 1 - C(n - c, k) / C(n, k), the chance that k samples drawn
    without replacement include at least one pass. Compute it so that
    n = 5000 does not overflow.

    Args:
        n: Samples drawn for the problem.
        c: Samples that passed, 0 <= c <= n.
        k: Attempts allowed, 1 <= k <= n.

    Returns:
        A float in [0, 1].
    """
    raise NotImplementedError("write pass_at_k")


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
    raise NotImplementedError("write bootstrap_ci")


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
    raise NotImplementedError("write paired_bootstrap_diff")


def ngram_overlap(train_docs: list, test_doc: list, n: int) -> float:
    """Fraction of the test document's n-grams found in any training document.

    Args:
        train_docs: A list of documents, each a list of tokens.
        test_doc: One document, a list of tokens.
        n: The n-gram length.

    Returns:
        A float in [0, 1]. 0.0 when the test document has fewer than n tokens.
    """
    raise NotImplementedError("write ngram_overlap")


if __name__ == "__main__":
    print("bpb:", bits_per_byte(40 * np.log(2.0), 40))
    print("raw pick:", choose_answer(np.array([-6.0, -9.0, -12.0]), np.array([3, 6, 12]), False))
    print("normalised pick:", choose_answer(np.array([-6.0, -9.0, -12.0]), np.array([3, 6, 12]), True))
    print("pass@2:", pass_at_k(5, 2, 2))
    rng = np.random.default_rng(0)
    print("ci:", bootstrap_ci((rng.uniform(size=100) < 0.6).astype(float), 500, rng))
    print("overlap:", ngram_overlap([list("xbcdy")], list("abcde"), 3))
