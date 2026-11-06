"""Reference solution: flops against bytes, and a key-value cache."""

import math

import numpy as np


def arithmetic_intensity(batch: int, bytes_per_elt: float) -> float:
    # Per weight: 2 * batch flops (a multiply and an add per vector), one read.
    return 2.0 * batch / bytes_per_elt


def decode_seconds_per_token(
    n_params: float, bytes_per_elt: float, bandwidth: float
) -> float:
    return n_params * bytes_per_elt / bandwidth


def min_batch_to_saturate(
    bytes_per_elt: float, peak_flops: float, bandwidth: float
) -> int:
    ridge = peak_flops / bandwidth
    # Solve 2 * B / b >= ridge for B, then round up to a whole batch. The tiny
    # offset keeps a ridge that is an exact integer from rounding up by one.
    return int(math.ceil(ridge * bytes_per_elt / 2.0 - 1e-9))

def kv_cache_bytes(
    n_layer: int, n_kv: int, d_head: int, seq_len: int, bytes_per_elt: int = 2
) -> int:
    # 2 for keys and values, then one number per layer, head, width, and token.
    return 2 * n_layer * n_kv * d_head * seq_len * bytes_per_elt


class KVCache:
    def __init__(self, n_layer: int, n_kv: int, d_head: int):
        self.n_kv = n_kv
        self.d_head = d_head
        self._keys = [[] for _ in range(n_layer)]
        self._values = [[] for _ in range(n_layer)]

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        self._keys[layer].append(np.asarray(k, dtype=float))
        self._values[layer].append(np.asarray(v, dtype=float))

    def get(self, layer: int) -> tuple:
        if not self._keys[layer]:
            empty = np.zeros((0, self.n_kv, self.d_head))
            return empty, empty.copy()
        return (
            np.stack(self._keys[layer], axis=0),
            np.stack(self._values[layer], axis=0),
        )


def cached_attention_step(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    scores = K @ q / np.sqrt(q.shape[0])
    exp = np.exp(scores - scores.max())
    weights = exp / exp.sum()
    return weights @ V


if __name__ == "__main__":
    n_params = 7e9
    b = 2
    peak = 312e12
    bw = 2e12

    print(f"intensity at batch 1: {arithmetic_intensity(1, b):.1f} flops per byte")
    print(f"ridge point: {peak / bw:.0f} flops per byte")
    t = decode_seconds_per_token(n_params, b, bw)
    print(f"one token reads the weights in {t * 1e3:.2f} ms ({1 / t:.0f} tokens per second)")
    print(f"batch needed to saturate the chip: {min_batch_to_saturate(b, peak, bw)}")

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
