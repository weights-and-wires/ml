"""Reference solution: decoding costs, a key-value cache, sampling, and speculative decoding."""

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


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / temperature
    exp = np.exp(scaled - scaled.max())
    return exp / exp.sum()


def _top_k(probs: np.ndarray, k: int) -> np.ndarray:
    keep = np.argsort(-probs, kind="stable")[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def _nucleus(probs: np.ndarray, p: float) -> np.ndarray:
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    # The first position whose running total reaches p is the last one kept.
    # Rounding can leave the final total just under 1.0, so clamp the index.
    cutoff = min(int(np.searchsorted(cumulative, p, side="left")), len(probs) - 1)
    out = np.zeros_like(probs)
    keep = order[: cutoff + 1]
    out[keep] = probs[keep]
    return out / out.sum()


def filter_probs(
    probs: np.ndarray, top_k: int | None = None, top_p: float | None = None
) -> np.ndarray:
    out = np.asarray(probs, dtype=float).copy()
    if top_k is not None:
        out = _top_k(out, top_k)
    if top_p is not None:
        out = _nucleus(out, top_p)
    return out


def sample_token(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    if temperature == 0:
        return int(np.argmax(logits))
    probs = filter_probs(apply_temperature(logits, temperature), top_k, top_p)
    return int(rng.choice(len(probs), p=probs))


def expected_accept_rate(p: np.ndarray, q: np.ndarray) -> float:
    # q[x] * min(1, p[x] / q[x]) is min(p[x], q[x]) wherever q[x] > 0.
    return float(np.minimum(np.asarray(p, float), np.asarray(q, float)).sum())


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    leftover = np.maximum(p - np.asarray(q, dtype=float), 0.0)
    total = leftover.sum()
    if total <= 0.0:
        return p.copy()
    return leftover / total


def speculative_accept(
    p: np.ndarray, q: np.ndarray, draft_token: int, rng: np.random.Generator
) -> int:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if rng.uniform() < min(1.0, p[draft_token] / q[draft_token]):
        return int(draft_token)
    residual = residual_distribution(p, q)
    return int(rng.choice(len(p), p=residual))


def beam_search(
    log_probs: np.ndarray,
    bos: int,
    eos: int,
    beam_width: int,
    max_len: int,
    alpha: float = 0.0,
) -> tuple:
    log_probs = np.asarray(log_probs, dtype=float)
    vocab = log_probs.shape[0]

    def normalised(raw: float, n_tokens: int) -> float:
        # n_tokens counts the generated tokens, so it excludes bos.
        return raw / ((5 + n_tokens) / 6) ** alpha

    beam = [([bos], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for tokens, raw in beam:
            for j in range(vocab):
                step = log_probs[tokens[-1], j]
                if step != -np.inf:
                    candidates.append((tokens + [j], raw + step))
        # A stable sort keeps beam order, then token order, among ties.
        candidates.sort(key=lambda c: -c[1])
        beam = []
        for tokens, raw in candidates[:beam_width]:
            if tokens[-1] == eos:
                finished.append((tokens, normalised(raw, len(tokens) - 1)))
            else:
                beam.append((tokens, raw))
        if not beam:
            break
    for tokens, raw in beam:
        finished.append((tokens, normalised(raw, len(tokens) - 1)))
    best = max(finished, key=lambda f: f[1])  # max keeps the first of equal scores
    return best[0], best[1]


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

    ln = np.log
    trap = np.full((4, 4), -np.inf)
    trap[0, 1:] = ln([0.5, 0.4, 0.1])
    trap[1, 1:] = ln([0.3, 0.3, 0.4])
    trap[2, 1:] = ln([0.05, 0.05, 0.9])
    trap[3, 3] = 0.0
    for width in (1, 2):
        tokens, score = beam_search(trap, bos=0, eos=3, beam_width=width, max_len=4)
        print(f"beam {width}: {tokens} with probability {np.exp(score):.2f}")
