"""A key-value cache: size it, fill it one token at a time, and read from it.

Fill in `kv_cache_bytes`, `KVCache`, and `cached_attention_step`. One layer's
cache holds, for every token so far, one key and one value per key-value head.
"""

import numpy as np


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


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    size = kv_cache_bytes(24, 8, 64, 2048, 2)
    print(f"24 layers, 8 kv heads, width 64, 2048 tokens: {size / 2**20:.1f} MiB")

    cache = KVCache(n_layer=1, n_kv=1, d_head=4)
    for t in range(3):
        cache.append(0, rng.normal(size=(1, 4)), rng.normal(size=(1, 4)))
    K, V = cache.get(0)
    print(f"after 3 tokens: K.shape={K.shape}, V.shape={V.shape}")

    q = rng.normal(size=4)
    out = cached_attention_step(q, K[:, 0, :], V[:, 0, :])
    print(f"attention output for the next query: {np.round(out, 3)}")
