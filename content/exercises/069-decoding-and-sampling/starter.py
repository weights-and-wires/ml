"""Decoding: flops against bytes, a key-value cache, sampling, speculation, and beam search.

Fill in the thirteen items below. For the first three, all quantities are plain
numbers: flops and bytes are counts, `bandwidth` is bytes per second, and
`peak_flops` is flops per second. For the cache, one layer holds, for every
token so far, one key and one value per key-value head. Then turn scores into a
token with temperature, top-k, and top-p, and write the accept-or-replace rule
that lets a draft model speed up a target model. Last, search for the most
probable whole sequence with a beam.
"""

import math

import numpy as np


def arithmetic_intensity(batch: int, bytes_per_elt: float) -> float:
    """Flops done per byte read when a weight matrix multiplies `batch` vectors.

    Every weight is read once. It then does one multiply and one add for each
    of the `batch` vectors. The width of the matrix does not appear.

    Args:
        batch: Number of vectors that share one read of the weights.
        bytes_per_elt: Bytes per weight, for example 2 for bf16.

    Returns:
        Flops per byte, as a float.
    """
    raise NotImplementedError("arithmetic_intensity")


def decode_seconds_per_token(
    n_params: float, bytes_per_elt: float, bandwidth: float
) -> float:
    """Seconds to read every weight once, which is the time of one decode step.

    This is the memory-bound estimate for batch 1. It ignores the cache and
    any arithmetic, so it is a lower bound on the true time.

    Args:
        n_params: Number of weights in the model.
        bytes_per_elt: Bytes per weight.
        bandwidth: Memory bandwidth in bytes per second.

    Returns:
        Seconds per token.
    """
    raise NotImplementedError("decode_seconds_per_token")


def min_batch_to_saturate(
    bytes_per_elt: float, peak_flops: float, bandwidth: float
) -> int:
    """Smallest whole batch whose arithmetic intensity reaches the ridge point.

    The ridge point is `peak_flops / bandwidth`, the flops per byte at which a
    chip stops waiting on memory and starts running out of arithmetic.

    Args:
        bytes_per_elt: Bytes per weight.
        peak_flops: Peak arithmetic rate in flops per second.
        bandwidth: Memory bandwidth in bytes per second.

    Returns:
        The smallest integer batch B with `arithmetic_intensity(B, bytes_per_elt)`
        at least the ridge point.
    """
    raise NotImplementedError("min_batch_to_saturate")

def kv_cache_bytes(
    n_layer: int, n_kv: int, d_head: int, seq_len: int, bytes_per_elt: int = 2
) -> int:
    """Bytes needed to cache the keys and values of one sequence.

    Args:
        n_layer: Number of transformer layers.
        n_kv: Number of key-value heads per layer.
        d_head: Width of one head.
        seq_len: Number of tokens cached.
        bytes_per_elt: Bytes per stored number, for example 2 for bf16.

    Returns:
        The total number of bytes, as an int. Keys and values are both counted.
    """
    raise NotImplementedError("kv_cache_bytes")


class KVCache:
    """Per-layer storage that grows by one row per token."""

    def __init__(self, n_layer: int, n_kv: int, d_head: int):
        """Create an empty cache for `n_layer` layers.

        Args:
            n_layer: Number of layers. Each keeps its own keys and values.
            n_kv: Key-value heads per layer.
            d_head: Width of one head.
        """
        raise NotImplementedError("KVCache.__init__")

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Store the key and value of one new token in `layer`.

        Args:
            layer: Which layer's cache to extend.
            k: The token's key, shape `(n_kv, d_head)`.
            v: The token's value, shape `(n_kv, d_head)`.
        """
        raise NotImplementedError("KVCache.append")

    def get(self, layer: int) -> tuple:
        """Return everything stored so far for `layer`.

        Returns:
            A pair `(K, V)`, each of shape `(T, n_kv, d_head)`, where `T` is the
            number of tokens appended to this layer. Row `t` is the token
            appended `t`-th. When nothing is stored, `T` is 0 and each array has
            shape `(0, n_kv, d_head)`.
        """
        raise NotImplementedError("KVCache.get")


def cached_attention_step(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Single-head attention of one new query over cached keys and values.

    Args:
        q: The new token's query, shape `(d_head,)`.
        K: Cached keys, shape `(T, d_head)`.
        V: Cached values, shape `(T, d_head)`.

    Returns:
        The attention output, shape `(d_head,)`: the weights
        `softmax(K @ q / sqrt(d_head))` blend the rows of `V`.
    """
    raise NotImplementedError("cached_attention_step")


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


def beam_search(
    log_probs: np.ndarray,
    bos: int,
    eos: int,
    beam_width: int,
    max_len: int,
    alpha: float = 0.0,
) -> tuple:
    """Beam search over a Markov model given as a transition log-probability matrix.

    `log_probs[i, j]` is the log-probability that token `j` follows token `i`;
    `-inf` means impossible. A hypothesis is the tokens after `bos`, and its
    raw score is the sum of the log-probabilities of its steps.

    At each of at most `max_len` steps, extend every live hypothesis with every
    token, skipping extensions whose score is `-inf`. Keep the `beam_width`
    best extensions over all of them together, not per hypothesis, and
    break ties by the order of the extension: live hypotheses in beam order,
    then tokens in id order. A kept extension that ends in `eos` is finished:
    it leaves the beam and is never extended again. After step `max_len`, the
    hypotheses still live are finished as they stand.

    A finished hypothesis of length `L` (counting `eos`, not `bos`) is ranked
    by `raw_score / ((5 + L) / 6) ** alpha`, the length penalty of Wu et al.
    (2016). With `alpha=0` that is the raw score.

    Args:
        log_probs: Transition log-probabilities, shape `(V, V)`.
        bos: Id of the start token.
        eos: Id of the end token.
        beam_width: Number of hypotheses kept at each step.
        max_len: Most tokens a hypothesis may have after `bos`.
        alpha: Length-penalty strength, 0 or larger.

    Returns:
        A pair `(tokens, score)`: the best finished hypothesis as a list of
        token ids that starts with `bos`, and its normalised score. When two
        finished hypotheses tie, the one that finished first wins.
    """
    raise NotImplementedError("beam_search")


if __name__ == "__main__":
    b, peak, bw = 2, 312e12, 2e12
    print(f"intensity at batch 1: {arithmetic_intensity(1, b):.1f} flops per byte")
    t = decode_seconds_per_token(7e9, b, bw)
    print(f"one token reads the weights in {t * 1e3:.2f} ms ({1 / t:.0f} tokens per second)")
    print(f"batch needed to saturate the chip: {min_batch_to_saturate(b, peak, bw)}")
    size = kv_cache_bytes(24, 8, 64, 2048, 2)
    print(f"24 layers, 8 kv heads, width 64, 2048 tokens: {size / 2**20:.1f} MiB")

    rng = np.random.default_rng(0)
    cache = KVCache(n_layer=1, n_kv=1, d_head=4)
    for _ in range(3):
        cache.append(0, rng.normal(size=(1, 4)), rng.normal(size=(1, 4)))
    K, V = cache.get(0)
    out = cached_attention_step(rng.normal(size=4), K[:, 0, :], V[:, 0, :])
    print(f"after 3 tokens: K.shape={K.shape}; attention output {np.round(out, 3)}")

    logits = np.array([0.5, -2.0, 3.5, 3.4, -1.0])
    for tau in (0.5, 1.0, 2.0):
        print(f"temperature {tau}: {np.round(apply_temperature(logits, tau), 4)}")
    probs = apply_temperature(logits, 2.0)
    print(f"top_p 0.9 at temperature 2: {np.round(filter_probs(probs, top_p=0.9), 4)}")
    draws = [sample_token(logits, rng, temperature=2.0, top_p=0.9) for _ in range(5000)]
    print(f"frequencies of 5000 draws: {np.round(np.bincount(draws, minlength=5) / 5000, 3)}")
    print(f"greedy choice: {sample_token(logits, rng, temperature=0)}")

    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])
    print(f"acceptance rate: {expected_accept_rate(p, q):.3f}")
    print(f"residual: {np.round(residual_distribution(p, q), 4)}")
    out = [speculative_accept(p, q, int(rng.choice(3, p=q)), rng) for _ in range(20_000)]
    print(f"output frequencies: {np.round(np.bincount(out, minlength=3) / len(out), 3)} (target {p})")
