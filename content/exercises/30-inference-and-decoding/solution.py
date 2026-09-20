"""Reference solution for the inference-and-decoding exercise."""

import numpy as np


def kv_cache_bytes(
    n_layer: int, n_kv: int, d_head: int, seq_len: int, bytes_per_elt: int = 2
) -> int:
    return 2 * n_layer * n_kv * d_head * seq_len * bytes_per_elt


class KVCache:
    def __init__(self, n_layer: int, n_kv: int, d_head: int):
        self.n_layer = n_layer
        self.n_kv = n_kv
        self.d_head = d_head
        self._k = [[] for _ in range(n_layer)]
        self._v = [[] for _ in range(n_layer)]

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        self._k[layer].append(np.asarray(k, dtype=float))
        self._v[layer].append(np.asarray(v, dtype=float))

    def get(self, layer: int) -> tuple:
        if not self._k[layer]:
            empty = np.zeros((0, self.n_kv, self.d_head))
            return empty, empty
        K = np.stack(self._k[layer], axis=0)
        V = np.stack(self._v[layer], axis=0)
        return K, V


def _stable_softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z)
    exp = np.exp(shifted)
    return exp / exp.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return _stable_softmax(np.asarray(logits, dtype=float) / temperature)


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    keep = np.argsort(-probs)[:k]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum()


def nucleus_filter(probs: np.ndarray, p: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, p, side="left"))
    cutoff = min(cutoff, len(probs) - 1)
    keep = order[: cutoff + 1]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    return filtered / filtered.sum()


def greedy_decode(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def speculative_accept(
    p: np.ndarray, q: np.ndarray, draft_token: int, rng: np.random.Generator
) -> int:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if q[draft_token] > 0:
        accept_prob = min(1.0, p[draft_token] / q[draft_token])
    else:
        accept_prob = 1.0
    if rng.uniform() < accept_prob:
        return int(draft_token)
    residual = np.maximum(p - q, 0.0)
    total = residual.sum()
    if total <= 0.0:
        residual, total = p, p.sum()
    residual = residual / total
    return int(rng.choice(len(p), p=residual))


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
