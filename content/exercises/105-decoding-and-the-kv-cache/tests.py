"""Checks for the flops-against-bytes and key-value cache exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


@check("arithmetic_intensity is 1 flop per byte for batch 1 in bf16")
def test_intensity_hand(sub):
    got = has(sub, "arithmetic_intensity")
    almost(got(1, 2), 1.0, tol=1e-12, label="arithmetic_intensity(1, 2)")
    almost(got(156, 2), 156.0, tol=1e-9, label="arithmetic_intensity(156, 2)")
    almost(got(8, 4), 4.0, tol=1e-12, label="arithmetic_intensity(8, 4)")


@check("decode_seconds_per_token reads 14 GB at 2 TB/s in 7 ms")
def test_decode_time(sub):
    got = has(sub, "decode_seconds_per_token")
    almost(got(7e9, 2, 2e12), 0.007, tol=1e-9, label="7B parameters in bf16")
    almost(got(7e9, 1, 2e12), 0.0035, tol=1e-9, label="7B parameters in 8-bit")
    almost(got(14e9, 2, 2e12), 0.014, tol=1e-9, label="14B parameters in bf16")


@check("min_batch_to_saturate gives 156 for the 312 TFLOPs, 2 TB/s chip in bf16")
def test_min_batch_hand(sub):
    got = has(sub, "min_batch_to_saturate")
    equal(got(2, 312e12, 2e12), 156, label="bf16")
    equal(got(4, 312e12, 2e12), 312, label="fp32")
    equal(got(1, 312e12, 2e12), 78, label="8-bit")


@check("The returned batch reaches the ridge and one fewer does not")
def test_min_batch_boundary(sub):
    intensity = has(sub, "arithmetic_intensity")
    got = has(sub, "min_batch_to_saturate")
    rng = np.random.default_rng(0)
    for _ in range(25):
        peak = rng.uniform(50, 900) * 1e12
        bw = rng.uniform(0.5, 4.0) * 1e12
        b = float(rng.choice([1, 2, 4]))
        ridge = peak / bw
        batch = got(b, peak, bw)
        if intensity(batch, b) < ridge:
            fail(f"batch {batch} has intensity {intensity(batch, b):.3f}, below the ridge {ridge:.3f}")
        if batch > 1 and intensity(batch - 1, b) >= ridge:
            fail(f"batch {batch - 1} already reaches the ridge {ridge:.3f}, so {batch} is not the smallest")


@check("kv_cache_bytes matches the hand count of 96 MiB")
def test_bytes_hand(sub):
    got = has(sub, "kv_cache_bytes")
    # 24 layers, 8 kv heads, width 64, 2048 tokens, 2 bytes per number.
    equal(int(got(24, 8, 64, 2048, 2)), 100_663_296, label="kv_cache_bytes(24, 8, 64, 2048, 2)")
    # One token of the same model is 48 KiB.
    equal(int(got(24, 8, 64, 1, 2)), 49_152, label="one token")


@check("KVCache.get returns the appended rows in order, shape (T, n_kv, d_head)")
def test_cache_basic(sub):
    cache_cls = has(sub, "KVCache")
    rng = np.random.default_rng(1)
    cache = cache_cls(n_layer=2, n_kv=2, d_head=3)
    tokens = [(rng.normal(size=(2, 3)), rng.normal(size=(2, 3))) for _ in range(4)]
    for k, v in tokens:
        cache.append(0, k, v)
    K, V = cache.get(0)
    shape(K, (4, 2, 3), label="K")
    shape(V, (4, 2, 3), label="V")
    close(K, np.stack([k for k, _ in tokens]), rtol=1e-9, atol=1e-12, label="cached keys")
    close(V, np.stack([v for _, v in tokens]), rtol=1e-9, atol=1e-12, label="cached values")


@check("An empty layer returns zero rows, and layers do not share storage")
def test_cache_empty_and_layers(sub):
    cache_cls = has(sub, "KVCache")
    rng = np.random.default_rng(2)
    cache = cache_cls(n_layer=2, n_kv=2, d_head=3)
    K, V = cache.get(1)
    shape(K, (0, 2, 3), label="empty K")
    shape(V, (0, 2, 3), label="empty V")
    cache.append(0, rng.normal(size=(2, 3)), rng.normal(size=(2, 3)))
    cache.append(0, rng.normal(size=(2, 3)), rng.normal(size=(2, 3)))
    K0, _ = cache.get(0)
    K1, _ = cache.get(1)
    shape(K0, (2, 2, 3), label="layer 0 after two appends")
    shape(K1, (0, 2, 3), label="layer 1 after appends to layer 0")


@check("cached_attention_step matches a manual softmax over the cached rows")
def test_step_manual(sub):
    step = has(sub, "cached_attention_step")
    rng = np.random.default_rng(3)
    d = 4
    K = rng.normal(size=(5, d))
    V = rng.normal(size=(5, d))
    q = rng.normal(size=d)
    weights = _softmax(np.array([K[t] @ q for t in range(5)]) / np.sqrt(d))
    want = sum(weights[t] * V[t] for t in range(5))
    out = step(q, K, V)
    shape(out, (d,), label="output")
    close(out, want, rtol=1e-8, atol=1e-10, label="cached_attention_step")


@check("With one cached row the output is that row's value, and large scores stay finite")
def test_step_edges(sub):
    step = has(sub, "cached_attention_step")
    rng = np.random.default_rng(4)
    V1 = rng.normal(size=(1, 3))
    close(step(rng.normal(size=3), rng.normal(size=(1, 3)), V1), V1[0], atol=1e-12, label="one row")
    q = np.array([1000.0, 0.0, 0.0])
    K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    V = np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    out = step(q, K, V)
    if not np.all(np.isfinite(out)):
        fail(f"cached_attention_step returned non-finite values {out} for a score near 577")
    close(out, V[0], atol=1e-9, label="the large score wins")


@check("Decoding token by token through KVCache equals a full causal recompute")
def test_incremental_equals_full(sub):
    cache_cls = has(sub, "KVCache")
    step = has(sub, "cached_attention_step")
    rng = np.random.default_rng(5)
    T, d_model, n_kv, d_head = 6, 8, 2, 4
    X = rng.normal(size=(T, d_model)) * 0.5
    Wq, Wk, Wv = (rng.normal(size=(d_model, n_kv * d_head)) * 0.4 for _ in range(3))

    # Full recompute: project every token, then a causal softmax per head.
    Q = (X @ Wq).reshape(T, n_kv, d_head)
    K = (X @ Wk).reshape(T, n_kv, d_head)
    V = (X @ Wv).reshape(T, n_kv, d_head)
    full = np.zeros((T, n_kv, d_head))
    for t in range(T):
        for h in range(n_kv):
            w = _softmax(K[: t + 1, h] @ Q[t, h] / np.sqrt(d_head))
            full[t, h] = w @ V[: t + 1, h]

    # Incremental: each step projects only the new token and reads the cache.
    cache = cache_cls(n_layer=1, n_kv=n_kv, d_head=d_head)
    inc = np.zeros((T, n_kv, d_head))
    for t in range(T):
        q_t = (X[t] @ Wq).reshape(n_kv, d_head)
        cache.append(0, (X[t] @ Wk).reshape(n_kv, d_head), (X[t] @ Wv).reshape(n_kv, d_head))
        Kc, Vc = cache.get(0)
        for h in range(n_kv):
            inc[t, h] = step(q_t[h], Kc[:, h], Vc[:, h])
    close(inc, full, rtol=1e-8, atol=1e-10, label="incremental vs full recompute")
