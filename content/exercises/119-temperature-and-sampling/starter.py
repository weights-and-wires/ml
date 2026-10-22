"""From scores to a token: temperature, filtering, and one random draw.

Fill in the three functions below. A vocabulary is a vector of `V` scores
(logits) or probabilities, one entry per token id.
"""

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide the logits by the temperature, then take a stable softmax.

    Args:
        logits: Scores, shape `(V,)`.
        temperature: A positive number. Below 1 sharpens the distribution,
            above 1 flattens it. It must survive values as small as 0.001.

    Returns:
        Probabilities of shape `(V,)`, positive and summing to 1.
    """
    raise NotImplementedError("apply_temperature")


def filter_probs(
    probs: np.ndarray, top_k: int | None = None, top_p: float | None = None
) -> np.ndarray:
    """Cut the tail of a distribution, then renormalise.

    Apply top-k first when `top_k` is given, keeping the `top_k` largest
    probabilities and renormalising. Then apply nucleus (top-p) filtering when
    `top_p` is given, on that renormalised distribution: keep the smallest set
    of largest probabilities whose total reaches `top_p`, including the token
    that makes the total reach it. The input is not sorted. With both arguments
    `None`, return the probabilities unchanged. The input is not modified.

    Args:
        probs: Probabilities, shape `(V,)`.
        top_k: Number of tokens to keep, or None.
        top_p: Mass to reach, in `(0, 1]`, or None. `top_p=1.0` must not fail.

    Returns:
        Probabilities of shape `(V,)`, zero outside the kept set, summing to 1.
    """
    raise NotImplementedError("filter_probs")


def sample_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Pick one token id from a vector of logits.

    A temperature of 0 means greedy decoding: return the index of the largest
    logit and use no randomness. Otherwise apply the temperature, filter with
    `filter_probs`, and draw one id with `rng.choice`.

    Args:
        logits: Scores, shape `(V,)`.
        rng: The random generator to draw from.
        temperature: Non-negative. 0 means greedy.
        top_k: Passed to `filter_probs`.
        top_p: Passed to `filter_probs`.

    Returns:
        The chosen token id as a plain Python int.
    """
    raise NotImplementedError("sample_token")


if __name__ == "__main__":
    logits = np.array([0.5, -2.0, 3.5, 3.4, -1.0])
    rng = np.random.default_rng(0)

    for tau in (0.5, 1.0, 2.0):
        print(f"temperature {tau}: {np.round(apply_temperature(logits, tau), 4)}")

    probs = apply_temperature(logits, 2.0)
    print(f"top_p 0.9 at temperature 2: {np.round(filter_probs(probs, top_p=0.9), 4)}")

    draws = [sample_token(logits, rng, temperature=2.0, top_p=0.9) for _ in range(5000)]
    print(f"frequencies of 5000 draws: {np.round(np.bincount(draws, minlength=5) / 5000, 3)}")
    print(f"greedy choice: {sample_token(logits, rng, temperature=0)}")
