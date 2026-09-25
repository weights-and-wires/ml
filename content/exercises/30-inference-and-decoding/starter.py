"""A KV cache, four ways to pick a token, and speculative decoding's rule.

Fill in the class and the six functions below. Each stands on its own; none
needs a real transformer, only the arrays that a transformer's layers would
produce.
"""

import numpy as np


def kv_cache_bytes(
    n_layer: int, n_kv: int, d_head: int, seq_len: int, bytes_per_elt: int = 2
) -> int:
    """Return the size, in bytes, of a KV cache holding `seq_len` tokens.

    Args:
        n_layer: Number of transformer layers.
        n_kv: Number of key-value heads (n_kv <= n_head; equal to n_head under
            ordinary multi-head attention, 1 under multi-query attention).
        d_head: Width of each attention head.
        seq_len: Number of cached positions.
        bytes_per_elt: Bytes per cached number (2 for bf16 or fp16).

    Returns:
        The total cache size in bytes, for one sequence.
    """
    raise NotImplementedError("kv_cache_bytes")


class KVCache:
    """A per-layer cache of key and value vectors, filled one token at a time."""

    def __init__(self, n_layer: int, n_kv: int, d_head: int):
        """Set up an empty cache for `n_layer` layers.

        Args:
            n_layer: Number of layers to cache.
            n_kv: Number of key-value heads per layer.
            d_head: Width of each head.
        """
        raise NotImplementedError("KVCache.__init__")

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append one new token's key and value for a given layer.

        Args:
            layer: Layer index, 0 <= layer < n_layer.
            k: The new token's key, shape (n_kv, d_head).
            v: The new token's value, shape (n_kv, d_head).
        """
        raise NotImplementedError("KVCache.append")

    def get(self, layer: int) -> tuple:
        """Return every cached key and value for a layer, in append order.

        Args:
            layer: Layer index.

        Returns:
            A tuple (K, V), each of shape (T, n_kv, d_head), where T is the
            number of tokens appended to this layer so far. T may be 0.
        """
        raise NotImplementedError("KVCache.get")


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return the softmax of logits / temperature.

    Args:
        logits: Unnormalised scores, shape (vocab,).
        temperature: A positive scalar. Small values sharpen the distribution
            toward the largest logit; large values flatten it toward uniform.

    Returns:
        A probability distribution, shape (vocab,), summing to 1.
    """
    raise NotImplementedError("apply_temperature")


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Zero out every probability outside the top k, and renormalise the rest.

    Args:
        probs: A probability distribution, shape (vocab,).
        k: Number of highest-probability entries to keep.

    Returns:
        A probability distribution, shape (vocab,), with exactly the top k
        entries of `probs` nonzero and rescaled to sum to 1.
    """
    raise NotImplementedError("top_k_filter")


def nucleus_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Zero out everything outside the nucleus, and renormalise what remains.

    The nucleus is the smallest set of tokens, taken in decreasing order of
    probability, whose cumulative probability is at least `p`.

    Steps:
        1. order = np.argsort(-probs) gives indices from most to least likely.
        2. Take np.cumsum of the sorted probabilities.
        3. Find the first position whose running total reaches p
           (np.searchsorted), and keep every index up to and including it.
        4. Copy the kept entries into a zero array and divide by their sum.

    Args:
        probs: A probability distribution, shape (vocab,).
        p: Cumulative probability threshold, in (0, 1].

    Returns:
        A probability distribution, shape (vocab,), with exactly the nucleus
        nonzero and rescaled to sum to 1.
    """
    raise NotImplementedError("nucleus_filter")


def greedy_decode(logits: np.ndarray) -> int:
    """Return the index of the largest logit.

    Args:
        logits: Unnormalised scores, shape (vocab,).

    Returns:
        The argmax index, as a plain int.
    """
    raise NotImplementedError("greedy_decode")


def speculative_accept(
    p: np.ndarray, q: np.ndarray, draft_token: int, rng: np.random.Generator
) -> int:
    """Accept or replace a draft model's proposed token.

    Accept `draft_token` with probability min(1, p[draft_token] /
    q[draft_token]). On rejection, return a token sampled from the residual
    distribution norm(max(0, p - q)).

    Args:
        p: The target model's distribution, shape (vocab,).
        q: The draft model's distribution, shape (vocab,), assumed to have
            q[draft_token] > 0.
        draft_token: The token the draft model proposed.
        rng: A seeded numpy random Generator, used for every random draw this
            function needs.

    Returns:
        The accepted token, as a plain int: `draft_token` on acceptance, or a
        token drawn from the residual distribution on rejection.
    """
    raise NotImplementedError("speculative_accept")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("KV cache size, 24 layers, 8 kv heads, d_head=64, 2048 tokens, bf16:")
    print(f"  {kv_cache_bytes(24, 8, 64, 2048, 2) / (1024 ** 2):.1f} MiB")

    cache = KVCache(n_layer=1, n_kv=2, d_head=4)
    for t in range(3):
        cache.append(0, rng.normal(size=(2, 4)), rng.normal(size=(2, 4)))
    K, V = cache.get(0)
    print(f"\ncache after 3 tokens: K.shape={K.shape}, V.shape={V.shape}")

    logits = rng.normal(size=10) * 3
    print(f"\nlogits: {np.round(logits, 2)}")
    print(f"greedy token: {greedy_decode(logits)}")
    for temperature in (0.01, 1.0, 100.0):
        probs = apply_temperature(logits, temperature)
        print(f"  T={temperature:>6}: top prob {probs.max():.4f}")

    probs = apply_temperature(logits, 1.0)
    print(f"\ntop-3 filter: {np.round(top_k_filter(probs, 3), 4)}")
    print(f"nucleus p=0.9: {np.round(nucleus_filter(probs, 0.9), 4)}")

    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])
    draws = [speculative_accept(p, q, rng.choice(3, p=q), rng) for _ in range(20_000)]
    empirical = np.bincount(draws, minlength=3) / len(draws)
    print(f"\ntarget p={p}, draft q={q}\nempirical output distribution: {empirical}")
