"""Speculative decoding: when to keep a drafted token and what replaces it.

Fill in the three functions below. `p` is the large (target) model's
distribution over the vocabulary and `q` is the small (draft) model's, both
for the same position and both of shape `(V,)`.
"""

import numpy as np


def expected_accept_rate(p: np.ndarray, q: np.ndarray) -> float:
    """Probability that a token drawn from `q` is accepted.

    A token `x` is drawn with probability `q[x]` and then accepted with
    probability `min(1, p[x] / q[x])`.

    Args:
        p: Target distribution, shape `(V,)`.
        q: Draft distribution, shape `(V,)`.

    Returns:
        The acceptance rate, a float in `[0, 1]`.
    """
    raise NotImplementedError("expected_accept_rate")


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The distribution a rejected token is replaced from.

    It is `max(p - q, 0)`, divided by its own sum so that it adds up to 1.
    When `p` equals `q` that sum is 0 and no rejection can happen, so return a
    copy of `p` instead of dividing by zero.

    Args:
        p: Target distribution, shape `(V,)`.
        q: Draft distribution, shape `(V,)`.

    Returns:
        A distribution of shape `(V,)` that sums to 1.
    """
    raise NotImplementedError("residual_distribution")


def speculative_accept(
    p: np.ndarray, q: np.ndarray, draft_token: int, rng: np.random.Generator
) -> int:
    """Accept the drafted token, or replace it, so the output follows `p`.

    Draw one uniform number from `rng`. Accept `draft_token` when it is below
    `min(1, p[draft_token] / q[draft_token])`. Otherwise draw a replacement from
    the residual distribution with `rng.choice`. The draft token was drawn from
    `q`, so `q[draft_token] > 0`.

    Args:
        p: Target distribution, shape `(V,)`.
        q: Draft distribution, shape `(V,)`.
        draft_token: The token the draft model proposed.
        rng: The random generator to draw from.

    Returns:
        The token id to keep, as a plain Python int.
    """
    raise NotImplementedError("speculative_accept")


if __name__ == "__main__":
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])
    rng = np.random.default_rng(0)

    print(f"acceptance rate: {expected_accept_rate(p, q):.3f}")
    print(f"residual: {np.round(residual_distribution(p, q), 4)}")

    draws = [speculative_accept(p, q, int(rng.choice(3, p=q)), rng) for _ in range(20_000)]
    print(f"output frequencies: {np.round(np.bincount(draws, minlength=3) / len(draws), 3)}")
    print(f"target p:           {p}")
