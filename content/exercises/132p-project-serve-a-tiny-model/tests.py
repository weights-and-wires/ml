"""Checks for the capstone project: serve a small language model."""

import numpy as np

from mlcheck import check, close, equal, fail, almost, has

PROMPT = "the driver returned a refund"


_CACHE = {}


def _tokeniser(sub, seed=0):
    key = (id(sub), seed)
    if key not in _CACHE:
        data = sub.make_data(seed)
        merges, vocab = sub.train_bpe(data["corpus"], sub.VOCAB_SIZE)
        _CACHE[key] = (data, merges, vocab)
    return _CACHE[key]


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _top_k_probs(logits, k):
    p = _softmax(np.asarray(logits, dtype=float))
    keep = np.argsort(-p, kind="stable")[:k]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / out.sum()


@check("train_bpe learns the hand-worked merges and never merges across a piece")
def test_train_bpe(sub):
    merges, vocab = sub.train_bpe("aaabdaaabac", 259)
    equal(len(merges), 3, "number of merges for vocab_size 259 is 259 - 256 = 3, so")
    if merges[0] != (97, 97):
        fail(f"the first merge should be (97, 97), 'a' with 'a', which occurs 4 times; got {merges[0]}")
    if merges[1] != (97, 98) or merges[2] != (256, 257):
        fail(f"after (97, 97) the counts tie between (97, 98) and (256, 97) at 2 each, and ties "
             f"go to the smaller pair, so the merges should be [(97, 97), (97, 98), (256, 257)]; "
             f"got {merges}")
    equal(vocab[256], b"aa")
    equal(vocab[258], b"aaab")
    _, merges, vocab = _tokeniser(sub)
    equal(len(merges), sub.VOCAB_SIZE - 256, "merge count")
    if b" the" not in vocab.values():
        fail("the corpus is full of ' the', so ' the' should be one of the learned tokens")
    for i in range(256, sub.VOCAB_SIZE):
        tok = vocab[i]
        if b" " in tok[1:]:
            fail(f"token {i} is {tok!r}: a merge crossed a piece boundary, so the "
                 f"pieces from the pre-tokeniser must be counted on their own")


@check("encode replays the merges in the order they were learned")
def test_encode(sub):
    merges = [(116, 104), (256, 101)]
    equal(sub.encode("the", merges), [257], "encode('the')")
    equal(sub.encode("then", merges), [257, 110], "encode('then')")
    merges2 = [(97, 97), (256, 97)]
    equal(sub.encode("aaab", merges2), [257, 98], "encode('aaab')")
    equal(sub.encode("", merges), [], "encode('')")
    _, merges, vocab = _tokeniser(sub)
    ids = sub.encode(PROMPT, merges)
    if not all(0 <= i < sub.VOCAB_SIZE for i in ids):
        fail(f"ids must lie in [0, {sub.VOCAB_SIZE}); got {ids}")
    if len(ids) >= len(PROMPT):
        fail(f"{len(PROMPT)} characters should compress below {len(PROMPT)} ids; got {len(ids)}")


@check("decode inverts encode exactly on strings the tokeniser never saw")
def test_round_trip(sub):
    _, merges, vocab = _tokeniser(sub)
    strings = [
        "", "a", "  leading and  double  spaces ", "line one\nline two\n\n\tTabbed",
        "Der Fahrer gab die Rückerstattung zurück.", "ड्राइवर ने पैसे लौटा दिए",
        "party \U0001F389 time \U0001F680", "order 4821 shipped 2026-10-26!",
        "The DRIVER, however, returned NOTHING...",
    ]
    rng = np.random.default_rng(3)
    alphabet = list("abcdefghij kl\né世界\U0001F389,.0123")
    for _ in range(40):
        strings.append("".join(rng.choice(alphabet, size=int(rng.integers(1, 40)))))
    for s in strings:
        back = sub.decode(sub.encode(s, merges), vocab)
        if back != s:
            fail(f"decode(encode(s)) != s for s = {s!r}; got {back!r}")
    try:
        out = sub.decode([200, 65], vocab)
    except UnicodeDecodeError:
        fail("decode raised on a lone continuation byte; generated ids can be any bytes, "
             "so decode must replace invalid UTF-8 instead of raising")
    equal(out, "�A", "decode of an invalid byte")


@check("fertility counts tokens per word, and it is higher outside the training domain")
def test_fertility(sub):
    merges = [(116, 104), (256, 101)]
    almost(sub.fertility("the cat", merges), 2.5, 1e-12, "fertility('the cat') is 5 ids over 2 words, so")
    data, merges, _ = _tokeniser(sub)
    inside = sub.fertility("the driver shipped the parcel on monday.", merges)
    outside = sub.fertility("ड्राइवर ने पैसे लौटा दिए", merges)
    german = sub.fertility("Der Fahrer gab die Rückerstattung zurück.", merges)
    if inside > 1.6:
        fail(f"in-domain fertility should be at most 1.6 tokens per word; got {inside:.2f}")
    if outside < 6 * inside or german < 2 * inside:
        fail(f"out-of-domain text should cost far more tokens per word: in-domain {inside:.2f}, "
             f"German {german:.2f}, Hindi {outside:.2f}")


@check("decode_step with a KV cache gives the same logits as the full forward pass")
def test_decode_step(sub):
    data, merges, _ = _tokeniser(sub)
    params = data["params"]
    ids = sub.encode(PROMPT, merges)
    want = sub.forward(params, ids)
    cache = sub.new_cache(params)
    got = []
    for n, t in enumerate(ids):
        got.append(sub.decode_step(params, t, cache))
        equal(cache.length, n + 1, f"cache length after {n + 1} tokens")
    got = np.stack(got)
    close(got, want, rtol=1e-9, atol=1e-9, label="cached logits vs full recompute")
    K, V = cache.get(0)
    equal(K.shape, (len(ids), params["n_head"], 16), "cached keys shape")
    # A cache must carry state: the same token after a different history differs.
    other = sub.new_cache(params)
    sub.decode_step(params, ids[0] + 1, other)
    differs = np.abs(sub.decode_step(params, ids[1], other)
                     - sub.decode_step(params, ids[1], sub.new_cache(params))).max()
    if differs < 1e-6:
        fail("the logits for a token ignore earlier tokens; decode_step must attend to the cache")


@check("next_token_probs and sample_token apply temperature, top-k, and top-p")
def test_sampling(sub):
    logits = np.array([0.5, -2.0, 3.5, 3.4, -1.0])
    p1 = sub.next_token_probs(logits, 1.0)
    close(p1, _softmax(logits), rtol=1e-9, atol=1e-12, label="temperature 1")
    close(sub.next_token_probs(logits, 0.5), _softmax(logits / 0.5), rtol=1e-9, atol=1e-12,
          label="temperature 0.5")
    close(sub.next_token_probs(np.array([1000.0, 999.0]), 1.0), _softmax(np.array([1000.0, 999.0])),
          label="large logits must not overflow")
    pk = sub.next_token_probs(logits, 1.0, top_k=2)
    close(pk, _top_k_probs(logits, 2), rtol=1e-9, atol=1e-12, label="top_k=2")
    # Temperature 2, top_p 0.9 keeps the three largest, as in the chapter.
    pp = sub.next_token_probs(logits, 2.0, top_p=0.9)
    probs2 = _softmax(logits / 2.0)
    order = np.argsort(-probs2)
    kept = np.zeros(5)
    kept[order[:3]] = probs2[order[:3]]
    close(pp, kept / kept.sum(), rtol=1e-9, atol=1e-12, label="temperature 2, top_p 0.9")
    equal(sub.next_token_probs(logits, 0.0), np.array([0, 0, 1.0, 0, 0]), "temperature 0 is one-hot")
    equal(sub.sample_token(logits, np.random.default_rng(0), temperature=0.0), 2, "greedy token")
    rng = np.random.default_rng(1)
    draws = np.array([sub.sample_token(logits, rng, 1.0, top_k=2) for _ in range(4000)])
    freq = np.bincount(draws, minlength=5) / 4000
    if freq[[0, 1, 4]].sum() > 0:
        fail(f"top_k=2 drew a token outside the two largest: frequencies {freq}")
    close(freq, _top_k_probs(logits, 2), atol=0.03, label="top-k sampling frequencies")
    a = [sub.sample_token(logits, np.random.default_rng(5)) for _ in range(3)]
    equal(a[0], a[1], "the same seed gives the same token")


@check("cached generation equals uncached generation, token for token")
def test_generate(sub):
    data, merges, _ = _tokeniser(sub)
    params = data["params"]
    ids = sub.encode(PROMPT, merges)
    # A reference loop built only from the given forward pass.
    seq = list(ids)
    for _ in range(12):
        seq.append(int(np.argmax(sub.forward(params, seq)[-1])))
    want = seq[len(ids):]
    cached = sub.generate(params, ids, 12, np.random.default_rng(0), use_cache=True)
    uncached = sub.generate(params, ids, 12, np.random.default_rng(0), use_cache=False)
    equal(cached, want, "greedy, cached")
    equal(uncached, want, "greedy, uncached")
    for seed in range(3):
        kw = dict(temperature=0.9, top_k=20, top_p=0.95)
        a = sub.generate(params, ids, 10, np.random.default_rng(seed), use_cache=True, **kw)
        b = sub.generate(params, ids, 10, np.random.default_rng(seed), use_cache=False, **kw)
        equal(a, b, f"sampled tokens for seed {seed}, cached against uncached")


@check("residual_distribution and speculative_round follow the accept-or-replace rule")
def test_round(sub):
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])
    close(sub.residual_distribution(p, q), [6 / 7, 1 / 7, 0.0], atol=1e-12, label="residual")
    close(sub.residual_distribution(p, p), p, atol=1e-12, label="residual when p equals q")

    table = np.random.default_rng(0).dirichlet(np.ones(5), size=5)
    fn = lambda toks: table[np.asarray(toks)]
    rng = np.random.default_rng(1)
    for _ in range(20):
        out, n_acc, n_chk = sub.speculative_round(fn, fn, [0, 1], 3, rng)
        equal(len(out), 4, "an identical draft is always accepted, so a round gives k + 1 tokens")
        equal((n_acc, n_chk), (3, 3), "accepted and checked counts for an identical draft")
    # A draft with disjoint support from the target is always rejected.
    P = np.tile(np.array([0.5, 0.5, 0, 0, 0]), (5, 1))
    Q = np.tile(np.array([0, 0, 0, 0.5, 0.5]), (5, 1))
    for _ in range(20):
        out, n_acc, n_chk = sub.speculative_round(lambda t: P[np.asarray(t)],
                                                  lambda t: Q[np.asarray(t)], [0], 3, rng)
        equal((len(out), n_acc, n_chk), (1, 0, 1), "a hopeless draft gives one token, none accepted")
        if out[0] not in (0, 1):
            fail(f"the replacement must come from the target's support {{0, 1}}; got {out[0]}")


@check("speculative sampling reproduces the target's distribution and reports its acceptance rate")
def test_speculative_distribution(sub):
    rng_t = np.random.default_rng(11)
    P = rng_t.dirichlet(np.ones(5) * 0.6, size=5)
    Q = rng_t.dirichlet(np.ones(5) * 0.6, size=5)
    tf = lambda toks: P[np.asarray(toks)]
    df = lambda toks: Q[np.asarray(toks)]
    rng = np.random.default_rng(0)
    n = 6000
    counts = np.zeros((5, 5))
    for _ in range(n):
        toks, stats = sub.speculative_generate(tf, df, [0], 2, 3, rng)
        counts[toks[0], toks[1]] += 1
    want = P[0][:, None] * P
    tv = 0.5 * np.abs(counts / n - want).sum()
    if tv > 0.05:
        fail(f"the joint distribution of the first two tokens is off: total variation {tv:.3f} "
             f"from the target's, expected under 0.05. A rule that replaces from p instead of the "
             f"leftover, or accepts every token, gives 0.1 or more.")
    # The acceptance rate: with no context dependence it equals sum(min(p, q)).
    p0 = np.array([0.45, 0.3, 0.15, 0.06, 0.04])
    q0 = np.array([0.2, 0.25, 0.1, 0.2, 0.25])
    alpha = np.minimum(p0, q0).sum()
    rng = np.random.default_rng(2)
    toks, stats = sub.speculative_generate(lambda t: np.tile(p0, (len(t), 1)),
                                           lambda t: np.tile(q0, (len(t), 1)), [0], 4000, 3, rng)
    equal(len(toks), 4000, "number of tokens returned")
    for key in ("accepted", "checked", "target_passes", "acceptance_rate"):
        if key not in stats:
            fail(f"stats is missing '{key}'; got keys {sorted(stats)}")
    almost(stats["acceptance_rate"], alpha, 0.03, "reported acceptance rate against sum(min(p, q))")
    if stats["target_passes"] >= 4000:
        fail("speculation should need fewer target passes than tokens")


@check("lora_linear and merge_lora reproduce the chapter's 2 by 2 example")
def test_lora_layer(sub):
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    close(sub.lora_linear(x, W, A, B, 2.0, 1), [[7.0, 1.0]], label="lora_linear")
    close(sub.merge_lora(W, A, B, 2.0, 1), [[3.0, 4.0], [0.0, 1.0]], label="merge_lora")
    rng = np.random.default_rng(0)
    W = rng.normal(size=(6, 8))
    A = rng.normal(size=(4, 8))
    B = rng.normal(size=(6, 4))
    x = rng.normal(size=(3, 8))
    close(sub.lora_linear(x, W, A, B, 8.0, 4), x @ sub.merge_lora(W, A, B, 8.0, 4).T,
          rtol=1e-10, atol=1e-10, label="split against merged, random layer")
    close(sub.lora_linear(x[0], W, A, B, 8.0, 4), x[0] @ sub.merge_lora(W, A, B, 8.0, 4).T,
          rtol=1e-10, atol=1e-10, label="a single row (1-D input)")


@check("merged and unmerged adapters give identical logits, and the base weights stay untouched")
def test_merged_logits(sub):
    data, merges, _ = _tokeniser(sub)
    params, adapter = data["params"], data["adapter"]
    before = [layer["Wq"].copy() for layer in params["layers"]]
    ids = sub.encode(PROMPT, merges)
    merged = sub.merge_adapter(params, adapter)
    for layer, w in zip(params["layers"], before):
        if not np.array_equal(layer["Wq"], w):
            fail("merge_adapter changed the base model's weights in place; it must return a copy")
    on_the_fly = sub.forward(params, ids, linear=sub.make_lora_linear(adapter))
    base = sub.forward(params, ids)
    close(sub.forward(merged, ids), on_the_fly, rtol=1e-8, atol=1e-8,
          label="merged against on-the-fly logits")
    if np.abs(on_the_fly - base).max() < 0.1:
        fail("the adapter should visibly change the logits; it changed them by "
             f"{np.abs(on_the_fly - base).max():.2g}")
    cache = sub.new_cache(merged)
    last = None
    for t in ids:
        last = sub.decode_step(merged, t, cache)
    cache2 = sub.new_cache(params)
    lin = sub.make_lora_linear(adapter)
    for t in ids:
        last2 = sub.decode_step(params, t, cache2, linear=lin)
    close(last, last2, rtol=1e-8, atol=1e-8, label="cached decoding, merged against on-the-fly")


@check("end to end: the served model answers the same merged or not, and speculation matches the target")
def test_end_to_end(sub):
    data, merges, vocab = _tokeniser(sub)
    params, adapter, draft = data["params"], data["adapter"], data["draft"]
    outs = {}
    for name, kw in (("base", {}), ("on the fly", dict(adapter=adapter)),
                     ("merged", dict(adapter=adapter, merge=True))):
        outs[name] = sub.complete(PROMPT, merges, vocab, params, 16, np.random.default_rng(0), **kw)
        if not outs[name].startswith(PROMPT):
            fail(f"the {name} completion should start with the prompt {PROMPT!r}; got {outs[name]!r}")
    equal(outs["merged"], outs["on the fly"], "greedy text, merged against on the fly")
    if outs["base"] == outs["merged"]:
        fail("the adapter should change what the model writes")
    s1 = sub.complete(PROMPT, merges, vocab, params, 16, np.random.default_rng(7),
                      temperature=0.8, top_k=30, adapter=adapter, merge=True)
    s2 = sub.complete(PROMPT, merges, vocab, params, 16, np.random.default_rng(7),
                      temperature=0.8, top_k=30, adapter=adapter)
    equal(s1, s2, "sampled text with the same seed, merged against on the fly")

    # Speculative decoding with a one-layer draft of the two-layer target.
    ids = sub.encode("the driver", merges)
    tf = sub.make_prob_fn(params, 1.0, 3)
    df = sub.make_prob_fn(draft, 1.0, 3)
    want = _top_k_probs(sub.forward(params, ids)[-1], 3)
    rng = np.random.default_rng(0)
    n = 1200
    counts = np.zeros(sub.VOCAB_SIZE)
    rates = []
    for _ in range(n):
        toks, stats = sub.speculative_generate(tf, df, ids, 1, 3, rng)
        counts[toks[0]] += 1
        rates.append(stats["acceptance_rate"])
    tv = 0.5 * np.abs(counts / n - want).sum()
    if tv > 0.06:
        fail(f"over {n} seeded runs the first token's distribution is {tv:.3f} away (total "
             f"variation) from the target's top-3 distribution {np.round(want[want > 0], 3)}; "
             f"expected under 0.06")
