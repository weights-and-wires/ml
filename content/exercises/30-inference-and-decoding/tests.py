"""Checks for the inference-and-decoding exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _full_recompute_attention(X, Wq, Wk, Wv, n_kv, d_head):
    """A full, non-cached causal self-attention forward pass, for comparison."""
    T, d_model = X.shape
    Q = (X @ Wq).reshape(T, n_kv, d_head)
    K = (X @ Wk).reshape(T, n_kv, d_head)
    V = (X @ Wv).reshape(T, n_kv, d_head)
    scale = 1.0 / np.sqrt(d_head)
    out = np.zeros((T, n_kv, d_head))
    for t in range(T):
        for h in range(n_kv):
            scores = (K[: t + 1, h, :] @ Q[t, h, :]) * scale
            weights = _softmax(scores)
            out[t, h] = weights @ V[: t + 1, h, :]
    return out.reshape(T, d_model)


def _incremental_attention(X, Wq, Wk, Wv, n_kv, d_head, cache_cls):
    """The same computation, one token at a time, through a KVCache."""
    T, d_model = X.shape
    cache = cache_cls(n_layer=1, n_kv=n_kv, d_head=d_head)
    scale = 1.0 / np.sqrt(d_head)
    out = np.zeros((T, n_kv, d_head))
    for t in range(T):
        x_t = X[t]
        q_t = (x_t @ Wq).reshape(n_kv, d_head)
        k_t = (x_t @ Wk).reshape(n_kv, d_head)
        v_t = (x_t @ Wv).reshape(n_kv, d_head)
        cache.append(0, k_t, v_t)
        K, V = cache.get(0)
        for h in range(n_kv):
            scores = (K[:, h, :] @ q_t[h]) * scale
            weights = _softmax(scores)
            out[t, h] = weights @ V[:, h, :]
    return out.reshape(T, d_model)


@check("kv_cache_bytes matches a hand-computed value")
def test_kv_cache_bytes_hand(sub):
    got = has(sub, "kv_cache_bytes")
    # 24 layers, 8 kv heads, d_head=64, 2048 tokens, bf16 (2 bytes/element).
    almost(
        got(24, 8, 64, 2048, 2),
        100_663_296,
        tol=1.0,
        label="kv_cache_bytes(24, 8, 64, 2048, 2)",
    )


@check("kv_cache_bytes scales linearly in n_kv, matching MQA/GQA's cache reduction")
def test_kv_cache_bytes_mqa(sub):
    got = has(sub, "kv_cache_bytes")
    full = got(n_layer=32, n_kv=8, d_head=128, seq_len=32_768, bytes_per_elt=2)
    mqa = got(n_layer=32, n_kv=1, d_head=128, seq_len=32_768, bytes_per_elt=2)
    if full != 8 * mqa:
        fail(
            f"dropping from 8 kv heads to 1 should shrink the cache by exactly "
            f"8x, but full={full:,} and mqa={mqa:,} (ratio {full / mqa:.3f})"
        )


@check("greedy_decode returns the index of the largest logit")
def test_greedy_decode(sub):
    got = has(sub, "greedy_decode")
    logits = np.array([0.5, -2.0, 3.5, 3.4, -1.0])
    if got(logits) != 2:
        fail(f"greedy_decode returned {got(logits)}, expected index 2")


@check("apply_temperature matches a manual softmax of logits / temperature")
def test_apply_temperature(sub):
    got = has(sub, "apply_temperature")
    logits = np.array([1.0, 3.0, 0.5, -1.0])
    temperature = 2.0
    scaled = logits / temperature
    want = np.exp(scaled - scaled.max())
    want = want / want.sum()
    probs = got(logits, temperature)
    close(probs, want, rtol=1e-8, atol=1e-10, label="apply_temperature")
    almost(float(probs.sum()), 1.0, tol=1e-9, label="probabilities summing to 1")


@check("apply_temperature at a small value converges to the greedy choice")
def test_apply_temperature_greedy_limit(sub):
    apply_temperature = has(sub, "apply_temperature")
    greedy_decode = has(sub, "greedy_decode")
    logits = np.array([1.0, 5.0, 2.0, 0.0, 4.5])
    probs = apply_temperature(logits, 1e-3)
    best = greedy_decode(logits)
    if probs[best] < 0.999:
        fail(
            f"at temperature 1e-3, the greedy token's probability should be "
            f"within 1e-3 of 1, but it is {probs[best]:.6f}"
        )


@check("top_k_filter keeps exactly the top k probabilities and renormalises")
def test_top_k_filter(sub):
    got = has(sub, "top_k_filter")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    filtered = got(probs, 2)
    want = np.array([0.4 / 0.7, 0.3 / 0.7, 0.0, 0.0])
    close(filtered, want, rtol=1e-9, atol=1e-12, label="top_k_filter(probs, 2)")
    almost(float(filtered.sum()), 1.0, tol=1e-9, label="renormalised mass")


@check("nucleus_filter keeps exactly the smallest set reaching p")
def test_nucleus_filter(sub):
    got = has(sub, "nucleus_filter")
    # Unsorted on purpose: the function must do its own sorting.
    probs = np.array([0.1, 0.5, 0.05, 0.2, 0.15])
    filtered = got(probs, 0.7)
    # Sorted descending: 0.5, 0.2, 0.15, 0.1, 0.05 with cumulative sums
    # 0.5, 0.7, 0.85, ...; the smallest prefix reaching 0.7 is {0.5, 0.2},
    # which are entries 1 and 3 of the original array.
    nonzero = np.flatnonzero(filtered)
    if sorted(nonzero.tolist()) != [1, 3]:
        fail(
            f"the nucleus at p=0.7 should be entries {{1, 3}} (probabilities "
            f"0.5 and 0.2), but the nonzero entries are {nonzero.tolist()}"
        )
    want = np.zeros(5)
    want[1] = 0.5 / 0.7
    want[3] = 0.2 / 0.7
    close(filtered, want, rtol=1e-9, atol=1e-12, label="nucleus_filter(probs, 0.7)")
    almost(float(filtered.sum()), 1.0, tol=1e-9, label="renormalised mass")


@check("KVCache.append and .get reproduce the appended keys and values in order")
def test_kv_cache_basic(sub):
    cache_cls = has(sub, "KVCache")
    rng = np.random.default_rng(1)
    cache = cache_cls(n_layer=2, n_kv=2, d_head=3)
    tokens = [(rng.normal(size=(2, 3)), rng.normal(size=(2, 3))) for _ in range(4)]
    for k, v in tokens:
        cache.append(0, k, v)
    K, V = cache.get(0)
    if K.shape != (4, 2, 3) or V.shape != (4, 2, 3):
        fail(f"KVCache.get returned shapes K={K.shape}, V={V.shape}, expected (4, 2, 3)")
    stacked_k = np.stack([k for k, _ in tokens], axis=0)
    stacked_v = np.stack([v for _, v in tokens], axis=0)
    close(K, stacked_k, rtol=1e-9, atol=1e-12, label="cached keys")
    close(V, stacked_v, rtol=1e-9, atol=1e-12, label="cached values")


@check("Incremental decoding through KVCache matches a full recompute")
def test_kv_cache_matches_full_recompute(sub):
    cache_cls = has(sub, "KVCache")
    rng = np.random.default_rng(3)
    n_kv, d_head, T = 3, 4, 6
    d_model = n_kv * d_head
    X = rng.normal(size=(T, d_model)) * 0.5
    Wq = rng.normal(size=(d_model, d_model)) * 0.3
    Wk = rng.normal(size=(d_model, d_model)) * 0.3
    Wv = rng.normal(size=(d_model, d_model)) * 0.3

    full = _full_recompute_attention(X, Wq, Wk, Wv, n_kv, d_head)
    incremental = _incremental_attention(X, Wq, Wk, Wv, n_kv, d_head, cache_cls)
    close(
        incremental,
        full,
        rtol=1e-8,
        atol=1e-10,
        label="attention output, incremental vs. full recompute",
    )


@check("speculative_accept reproduces the target distribution even with a bad draft")
def test_speculative_accept(sub):
    got = has(sub, "speculative_accept")
    rng = np.random.default_rng(7)
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])  # deliberately built to disagree with p

    n_samples = 20_000
    draft_tokens = rng.choice(3, size=n_samples, p=q)
    outputs = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        outputs[i] = got(p, q, int(draft_tokens[i]), rng)

    empirical = np.bincount(outputs, minlength=3) / n_samples
    close(
        empirical,
        p,
        atol=0.02,
        label="empirical output distribution vs. the target p",
    )
