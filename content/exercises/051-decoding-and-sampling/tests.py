"""Checks for the decoding, sampling, and speculative decoding exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape

LOGITS = np.array([0.5, -2.0, 3.5, 3.4, -1.0])
P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.1, 0.1, 0.8])


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


@check("arithmetic_intensity is 1 flop per byte at batch 1 in bf16, and decode reads 14 GB at 2 TB/s in 7 ms")
def test_memory_bound(sub):
    ai = has(sub, "arithmetic_intensity")
    almost(ai(1, 2), 1.0, tol=1e-12, label="arithmetic_intensity(1, 2)")
    almost(ai(156, 2), 156.0, tol=1e-9, label="arithmetic_intensity(156, 2)")
    almost(ai(8, 4), 4.0, tol=1e-12, label="arithmetic_intensity(8, 4)")
    dt = has(sub, "decode_seconds_per_token")
    almost(dt(7e9, 2, 2e12), 0.007, tol=1e-9, label="7B parameters in bf16")
    almost(dt(7e9, 1, 2e12), 0.0035, tol=1e-9, label="7B parameters in 8-bit")
    almost(dt(14e9, 2, 2e12), 0.014, tol=1e-9, label="14B parameters in bf16")


@check("min_batch_to_saturate gives 156 for the 312 TFLOPs chip, and one fewer does not reach the ridge")
def test_min_batch(sub):
    got = has(sub, "min_batch_to_saturate")
    equal(got(2, 312e12, 2e12), 156, label="bf16")
    equal(got(4, 312e12, 2e12), 312, label="fp32")
    equal(got(1, 312e12, 2e12), 78, label="8-bit")
    intensity = has(sub, "arithmetic_intensity")
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


@check("KVCache.get returns the appended rows in order, and an empty layer returns zero rows")
def test_cache(sub):
    cache_cls = has(sub, "KVCache")
    rng = np.random.default_rng(1)
    cache = cache_cls(n_layer=2, n_kv=2, d_head=3)
    K, V = cache.get(1)
    shape(K, (0, 2, 3), label="empty K")
    shape(V, (0, 2, 3), label="empty V")
    tokens = [(rng.normal(size=(2, 3)), rng.normal(size=(2, 3))) for _ in range(4)]
    for k, v in tokens:
        cache.append(0, k, v)
    K, V = cache.get(0)
    shape(K, (4, 2, 3), label="K")
    shape(V, (4, 2, 3), label="V")
    close(K, np.stack([k for k, _ in tokens]), rtol=1e-9, atol=1e-12, label="cached keys")
    close(V, np.stack([v for _, v in tokens]), rtol=1e-9, atol=1e-12, label="cached values")
    K1, _ = cache.get(1)
    shape(K1, (0, 2, 3), label="layer 1 after appends to layer 0")


@check("cached_attention_step matches a manual softmax, stays finite, and decoding through the cache equals a full recompute")
def test_step(sub):
    step = has(sub, "cached_attention_step")
    cache_cls = has(sub, "KVCache")
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
    V1 = rng.normal(size=(1, 3))
    close(step(rng.normal(size=3), rng.normal(size=(1, 3)), V1), V1[0], atol=1e-12, label="one row")
    big_q = np.array([1000.0, 0.0, 0.0])
    Kb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Vb = np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    out = step(big_q, Kb, Vb)
    if not np.all(np.isfinite(out)):
        fail(f"cached_attention_step returned non-finite values {out} for a score near 577")
    close(out, Vb[0], atol=1e-9, label="the large score wins")

    T, d_model, n_kv, d_head = 6, 8, 2, 4
    rng = np.random.default_rng(5)
    X = rng.normal(size=(T, d_model)) * 0.5
    Wq, Wk, Wv = (rng.normal(size=(d_model, n_kv * d_head)) * 0.4 for _ in range(3))
    Q = (X @ Wq).reshape(T, n_kv, d_head)
    Kf = (X @ Wk).reshape(T, n_kv, d_head)
    Vf = (X @ Wv).reshape(T, n_kv, d_head)
    full = np.zeros((T, n_kv, d_head))
    for t in range(T):
        for h in range(n_kv):
            w = _softmax(Kf[: t + 1, h] @ Q[t, h] / np.sqrt(d_head))
            full[t, h] = w @ Vf[: t + 1, h]
    cache = cache_cls(n_layer=1, n_kv=n_kv, d_head=d_head)
    inc = np.zeros((T, n_kv, d_head))
    for t in range(T):
        q_t = (X[t] @ Wq).reshape(n_kv, d_head)
        cache.append(0, (X[t] @ Wk).reshape(n_kv, d_head), (X[t] @ Wv).reshape(n_kv, d_head))
        Kc, Vc = cache.get(0)
        for h in range(n_kv):
            inc[t, h] = step(q_t[h], Kc[:, h], Vc[:, h])
    close(inc, full, rtol=1e-8, atol=1e-10, label="incremental vs full recompute")


@check("apply_temperature gives the hand-worked probabilities, stays finite when cold, and flattens when hot")
def test_temperature(sub):
    got = has(sub, "apply_temperature")
    close(got(LOGITS, 1.0), [0.0253, 0.0021, 0.5077, 0.4594, 0.0056], atol=6e-5, label="temperature 1")
    close(got(LOGITS, 0.5), [0.0014, 0.0000, 0.5490, 0.4495, 0.0001], atol=6e-5, label="temperature 0.5")
    close(got(LOGITS, 2.0), [0.0952, 0.0273, 0.4267, 0.4059, 0.0450], atol=6e-5, label="temperature 2")
    cold = got(LOGITS, 0.001)
    if not np.all(np.isfinite(cold)):
        fail(f"temperature 0.001 returned non-finite values {cold}; subtract the max before exp")
    close(cold, [0, 0, 1, 0, 0], atol=1e-12, label="temperature 0.001 is greedy")
    hot = got(LOGITS, 1000.0)
    close(hot, np.full(5, 0.2), atol=2e-3, label="temperature 1000 is nearly uniform")
    almost(float(hot.sum()), 1.0, tol=1e-12, label="probabilities sum to 1")
    taus = [0.25, 0.5, 1.0, 2.0, 4.0]
    tops = [got(LOGITS, t).max() for t in taus]
    tails = [got(LOGITS, t)[[0, 1, 4]].sum() for t in taus]
    for i in range(len(taus) - 1):
        if not tops[i] > tops[i + 1]:
            fail(f"top probability should fall from temperature {taus[i]} to {taus[i + 1]}, got {tops[i]:.4f} then {tops[i + 1]:.4f}")
        if not tails[i] < tails[i + 1]:
            fail(f"junk mass should rise from temperature {taus[i]} to {taus[i + 1]}, got {tails[i]:.4f} then {tails[i + 1]:.4f}")


@check("filter_probs applies top-k, top-p with the crossing token, p = 1.0, and top-k before top-p")
def test_filters(sub):
    got = has(sub, "filter_probs")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    close(got(probs, top_k=2), [4 / 7, 3 / 7, 0, 0], atol=1e-12, label="top_k=2")
    close(got(np.array([0.1, 0.35, 0.05, 0.5]), top_k=2), [0, 0.35 / 0.85, 0, 0.5 / 0.85], atol=1e-12, label="unsorted top_k=2")
    unchanged = np.array([0.5, 0.3, 0.2])
    close(got(unchanged, top_k=3), unchanged, atol=1e-12, label="top_k equal to the vocabulary size")
    pp = np.array([0.1, 0.5, 0.05, 0.2, 0.15])
    close(got(pp, top_p=0.7), [0, 0.5 / 0.7, 0, 0.2 / 0.7, 0], atol=1e-12, label="top_p=0.7")
    # The top token alone already exceeds p, and it must still be kept.
    close(got(pp, top_p=0.3), [0, 1, 0, 0, 0], atol=1e-12, label="top_p=0.3 keeps the leader")
    close(got(pp, top_p=0.8), [0, 0.5 / 0.85, 0, 0.2 / 0.85, 0.15 / 0.85], atol=1e-12, label="top_p=0.8")
    # The running total of ten 0.1s ends at 0.9999999999999999, not 1.0.
    flat = np.full(10, 0.1)
    close(got(flat, top_p=1.0), flat, atol=1e-12, label="top_p=1.0")
    original = flat.copy()
    got(flat, top_k=3, top_p=0.5)
    close(flat, original, atol=0, label="the input array is not modified")
    close(got(flat), flat, atol=1e-15, label="no filters")
    # Top-3 gives [4/9, 3/9, 2/9]; its running totals are 0.444, 0.778, so p=0.75 needs two tokens.
    close(got(probs, top_k=3, top_p=0.75), [4 / 7, 3 / 7, 0, 0], atol=1e-12, label="top_k=3, top_p=0.75")


@check("sample_token is greedy at temperature 0, returns plain ints, and draws from the filtered distribution only")
def test_sample(sub):
    got = has(sub, "sample_token")
    probs = has(sub, "apply_temperature")
    filt = has(sub, "filter_probs")
    rng = np.random.default_rng(0)
    out = got(LOGITS, rng, temperature=0)
    equal(out, 2, label="greedy choice")
    if type(out) is not int:
        fail(f"sample_token returned {type(out).__name__}, expected a plain int")
    draw = got(LOGITS, rng, temperature=1.0)
    if type(draw) is not int:
        fail(f"sample_token returned {type(draw).__name__} for a random draw, expected a plain int")
    rng = np.random.default_rng(7)
    n = 10_000
    draws = np.array([got(LOGITS, rng, temperature=2.0, top_p=0.9) for _ in range(n)])
    want = filt(probs(LOGITS, 2.0), top_p=0.9)
    freq = np.bincount(draws, minlength=5) / n
    if np.any(freq[want == 0] > 0):
        fail(f"drew a token that the nucleus removed; frequencies {np.round(freq, 3)}")
    close(freq, want, atol=0.02, label="empirical frequencies vs filtered probabilities")
    top2 = np.array([got(LOGITS, rng, temperature=1.0, top_k=2) for _ in range(2000)])
    if not set(np.unique(top2)) <= {2, 3}:
        fail(f"top_k=2 should only return ids 2 and 3, got {sorted(set(top2.tolist()))}")


@check("expected_accept_rate is the overlap of p and q, and residual_distribution is max(p - q, 0) normalised")
def test_rate_and_residual(sub):
    rate = has(sub, "expected_accept_rate")
    res = has(sub, "residual_distribution")
    almost(rate(P, Q), 0.3, tol=1e-12, label="rate for p=[.7,.2,.1], q=[.1,.1,.8]")
    almost(rate(P, P), 1.0, tol=1e-12, label="rate when the draft equals the target")
    almost(rate(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0, tol=1e-12, label="rate for disjoint distributions")
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        almost(rate(p, q), rate(q, p), tol=1e-12, label="symmetry")
        almost(rate(p, q), 1.0 - np.maximum(p - q, 0).sum(), tol=1e-12, label="1 minus the residual mass")
    close(res(P, Q), [6 / 7, 1 / 7, 0.0], atol=1e-12, label="residual for the worked example")
    out = res(np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.2, 0.6]))
    close(out, [0.3 / 0.4, 0.1 / 0.4, 0.0], atol=1e-12, label="residual for a second pair")
    almost(float(out.sum()), 1.0, tol=1e-12, label="sums to 1")
    same = res(P, P)
    if not np.all(np.isfinite(same)):
        fail(f"residual_distribution(p, p) returned {same}; it must not divide by zero")
    close(same, P, atol=1e-12, label="residual of identical distributions")


@check("speculative_accept keeps tokens with p >= q, replaces from the residual, and its output follows p")
def test_speculative(sub):
    got = has(sub, "speculative_accept")
    rng = np.random.default_rng(1)
    for token in (0, 1):
        for _ in range(200):
            out = got(P, Q, token, rng)
            if out != token:
                fail(f"draft token {token} has p >= q, so it must always be accepted, but got {out}")
    same = [got(P, P, 1, rng) for _ in range(200)]
    if any(t != 1 for t in same):
        fail("with q equal to p every draft token must be accepted")
    rng = np.random.default_rng(2)
    n = 20_000
    # Draft token 2: accepted with probability 0.125, else replaced from [6/7, 1/7, 0].
    # Output: token 0 with 0.875 * 6/7 = 0.75, token 1 with 0.125, token 2 with 0.125.
    draws = np.array([got(P, Q, 2, rng) for _ in range(n)])
    close(np.bincount(draws, minlength=3) / n, [0.75, 0.125, 0.125], atol=0.015, label="outputs given draft token 2")
    rng = np.random.default_rng(7)
    drafts = rng.choice(3, size=n, p=Q)
    out = np.array([got(P, Q, int(d), rng) for d in drafts])
    close(np.bincount(out, minlength=3) / n, P, atol=0.02, label="empirical output vs target p")
    q2 = np.array([0.25, 0.25, 0.5])
    drafts = rng.choice(3, size=n, p=q2)
    out = np.array([got(P, q2, int(d), rng) for d in drafts])
    close(np.bincount(out, minlength=3) / n, P, atol=0.02, label="empirical output vs p, milder draft")
