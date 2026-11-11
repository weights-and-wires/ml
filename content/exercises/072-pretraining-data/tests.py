"""Checks for the pretraining-data exercise.

Wrong implementations these checks are written to catch:

1. A bottom-k sketch passed off as a MinHash signature: one hash function, the
   k smallest values sorted, compared by position. It fails
   `test_signature_union_rule` (the signature of a union is not the elementwise
   minimum of the two signatures) and `test_estimate_error_shrinks`.
2. The banding probability written as (1 - (1 - s)**r)**b instead of
   1 - (1 - s**r)**b. It fails `test_lsh_probability`.
"""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


def _words(rng, vocab, n):
    return [vocab[i] for i in rng.integers(0, len(vocab), size=n)]


def _planted_corpus():
    """Return (docs, clusters). Clusters 0..3 hold 3 near-copies each; the rest are unique."""
    rng = np.random.default_rng(7)
    vocab = [f"w{i}" for i in range(2000)]
    docs, labels = [], []
    for c in range(4):
        base = _words(rng, vocab, 200)
        docs.append(" ".join(base))
        labels.append(c)
        for _ in range(2):
            copy = list(base)
            for pos in rng.choice(200, size=4, replace=False):
                copy[pos] = vocab[int(rng.integers(0, len(vocab)))]
            docs.append(" ".join(copy))
            labels.append(c)
    for u in range(8):
        docs.append(" ".join(_words(rng, vocab, 200)))
        labels.append(100 + u)
    order = rng.permutation(len(docs))
    return [docs[i] for i in order], [labels[i] for i in order]


@check("shingles cuts text into lowercase word k-grams")
def test_shingles(sub):
    fn = has(sub, "shingles")
    got = fn("The cat  sat on the mat", 2)
    want = {"the cat", "cat sat", "sat on", "on the", "the mat"}
    if set(got) != want:
        fail(f"expected {sorted(want)}, got {sorted(got)}")
    short = fn("hello world", 3)
    if set(short) != {"hello world"}:
        fail(f"a text shorter than k words is one shingle; expected {{'hello world'}}, got {short}")
    if len(fn("a b a b a b", 2)) != 2:
        fail(f"repeated shingles count once; expected 2 distinct, got {fn('a b a b a b', 2)}")
    if len(fn("", 2)) != 0:
        fail(f"empty text has no shingles; got {fn('', 2)}")


@check("jaccard is intersection over union: {a,b,c,d} and {c,d,e} give 2/5")
def test_jaccard(sub):
    fn = has(sub, "jaccard")
    almost(fn({"a", "b", "c", "d"}, {"c", "d", "e"}), 0.4, 1e-12, label="jaccard of the worked example")
    almost(fn({"a"}, {"a"}), 1.0, 1e-12, label="identical sets")
    almost(fn({"a"}, {"b"}), 0.0, 1e-12, label="disjoint sets")
    a = sub.shingles("the cat sat on the mat", 2)
    b = sub.shingles("the cat sat on the rug", 2)
    almost(fn(a, b), 4 / 6, 1e-12, label="the lesson's two sentences")


@check("exact_dedup keeps the first copy and ignores case and spacing")
def test_exact_dedup(sub):
    fn = has(sub, "exact_dedup")
    docs = ["Hello world", "other text", "hello   WORLD", "Hello world", "third"]
    got = fn(docs)
    equal(list(got), [0, 1, 4], label="kept indices")


@check("minhash_signature is seeded, has k entries, and the signature of a union is the elementwise minimum")
def test_signature_union_rule(sub):
    fn = has(sub, "minhash_signature")
    a = {f"a{i}" for i in range(30)}
    b = {f"b{i}" for i in range(30)}
    s1 = fn(a, 64, 3)
    if np.asarray(s1).shape != (64,):
        fail(f"expected shape (64,), got {np.asarray(s1).shape}")
    equal(np.asarray(fn(a, 64, 3)), np.asarray(s1), label="same seed, same signature")
    if np.array_equal(np.asarray(fn(a, 64, 4)), np.asarray(s1)):
        fail("a different seed returned the same signature")
    union = np.asarray(fn(a | b, 64, 3))
    want = np.minimum(np.asarray(fn(a, 64, 3)), np.asarray(fn(b, 64, 3)))
    if not np.array_equal(union, want):
        fail(
            "signature(A | B) must equal the elementwise minimum of signature(A) and signature(B); "
            f"got {union[:5]}..., expected {want[:5]}... (a bottom-k sketch compared by position breaks this rule)"
        )


@check("estimate_jaccard is exact at 1 and 0, and its error shrinks like 1 / sqrt(k)")
def test_estimate_error_shrinks(sub):
    est = has(sub, "estimate_jaccard")
    sig = has(sub, "minhash_signature")
    x = {f"x{i}" for i in range(20)}
    almost(est(sig(x, 64, 0), sig(x, 64, 0)), 1.0, 1e-12, label="identical sets")
    almost(est(sig(x, 64, 0), sig({f"y{i}" for i in range(20)}, 64, 0)), 0.0, 0.05, label="disjoint sets")

    def rms(k):
        errs = []
        for t in range(150):
            a = {f"p{t}-{i}" for i in range(60)}
            b = {f"p{t}-{i}" for i in range(20, 80)}  # 40 shared of 80 total: J = 0.5
            errs.append(est(sig(a, k, t), sig(b, k, t)) - 0.5)
        return float(np.sqrt(np.mean(np.square(errs))))

    small, large = rms(32), rms(512)
    want_small = np.sqrt(0.25 / 32)
    want_large = np.sqrt(0.25 / 512)
    almost(small, want_small, 0.25 * want_small, label="rms error at k = 32 (theory sqrt(0.25 / k))")
    almost(large, want_large, 0.25 * want_large, label="rms error at k = 512 (theory sqrt(0.25 / k))")
    ratio = small / large
    if not 3.0 < ratio < 5.5:
        fail(f"16 times more hashes must cut the error about 4 times; got ratio {ratio:.2f}")


@check("lsh_candidate_prob is 1 - (1 - s^r)^b and matches a Monte Carlo of the banding")
def test_lsh_probability(sub):
    fn = has(sub, "lsh_candidate_prob")
    almost(float(fn(0.5, 4, 5)), 1 - (1 - 0.5**4) ** 5, 1e-12, label="s = 0.5, r = 4, b = 5 (want 0.2758)")
    almost(float(fn(0.0, 4, 5)), 0.0, 1e-12, label="s = 0")
    almost(float(fn(1.0, 4, 5)), 1.0, 1e-12, label="s = 1")
    rng = np.random.default_rng(0)
    for s, r, b in [(0.8, 5, 20), (0.5, 3, 10)]:
        match = rng.random((20000, b, r)) < s  # each row of each band matches with probability s
        mc = float(np.mean(match.all(axis=2).any(axis=1)))
        almost(float(fn(s, r, b)), mc, 0.015, label=f"closed form against Monte Carlo at s={s}, r={r}, b={b}")
    out = np.asarray(fn(np.array([0.2, 0.5, 0.8]), 4, 5))
    if out.shape != (3,):
        fail(f"an array of similarities must return an array of shape (3,), got {out.shape}")


@check("lsh_candidates pairs the documents that agree on every row of at least one band")
def test_lsh_candidates(sub):
    fn = has(sub, "lsh_candidates")
    sigs = np.array(
        [
            [1, 2, 3, 4],  # doc 0
            [1, 2, 9, 9],  # doc 1 agrees with doc 0 on band 0 (columns 0 and 1)
            [7, 7, 3, 4],  # doc 2 agrees with doc 0 on band 1 (columns 2 and 3)
            [5, 6, 8, 8],  # doc 3 agrees with nobody on a whole band
            [1, 9, 3, 9],  # doc 4 matches doc 0 in two columns, but never both columns of one band
        ]
    )
    got = {tuple(sorted(p)) for p in fn(sigs, 2, 2)}
    want = {(0, 1), (0, 2)}
    if got != want:
        fail(f"expected candidate pairs {sorted(want)}, got {sorted(got)}")


@check("dedup keeps exactly one document from each planted near-duplicate cluster")
def test_dedup(sub):
    fn = has(sub, "dedup")
    docs, labels = _planted_corpus()
    kept = list(fn(docs))
    kept_labels = [labels[i] for i in kept]
    if len(set(kept_labels)) != len(kept_labels):
        fail(f"two documents from one cluster survived: kept labels {sorted(kept_labels)}")
    expected = len(set(labels))
    if len(kept) != expected:
        fail(f"expected {expected} survivors (4 clusters and 8 unique documents), got {len(kept)}")
    for c in range(4):
        members = [i for i, lab in enumerate(labels) if lab == c]
        if min(members) not in kept:
            fail(f"cluster {c}: expected the earliest document {min(members)} to survive, kept {sorted(kept)}")


@check("mix_weights and epochs_per_domain: T = 1 is proportional, T = 2 takes square roots, large T is uniform")
def test_mix(sub):
    mw = has(sub, "mix_weights")
    close(mw([900, 90, 10], 1.0), np.array([0.9, 0.09, 0.01]), atol=1e-12, label="T = 1")
    root = np.sqrt([900.0, 90.0, 10.0])
    close(mw([900, 90, 10], 2.0), root / root.sum(), atol=1e-12, label="T = 2 (0.7034, 0.2224, 0.0741)")
    close(mw([900, 90, 10], 1e6), np.full(3, 1 / 3), atol=1e-4, label="T = 1e6")
    close(mw([5.0, 0.0, 5.0], 2.0), np.array([0.5, 0.0, 0.5]), atol=1e-12, label="an empty domain gets weight 0")
    ep = has(sub, "epochs_per_domain")
    w = mw([900, 90, 10], 2.0)
    got = ep([900, 90, 10], w, 300)
    close(got, np.array([0.2345, 0.7413, 2.2243]), atol=1e-3, label="epochs of each domain at T = 2, budget 300")


def _clean_text(rng, n_sentences):
    subj = ["the model", "a student", "the teacher", "our team", "the engineer", "a small dog", "the old library"]
    verb = ["reads", "writes", "builds", "tests", "finds", "keeps", "checks", "shares"]
    obj = ["a short report", "the final answer", "every new idea", "some useful data", "the main result", "a clear plan"]
    tail = ["before the meeting.", "after the long lunch.", "with great care.", "on a quiet morning.", "for the whole class."]
    out = []
    for _ in range(n_sentences):
        out.append(f"{subj[rng.integers(7)]} {verb[rng.integers(8)]} {obj[rng.integers(6)]} {tail[rng.integers(5)]}")
    return " ".join(out)


def _noisy_text(rng, kind):
    printable = [chr(c) for c in range(33, 127)]
    if kind == 0:  # random printable characters
        return "".join(printable[i] for i in rng.integers(0, len(printable), size=160))
    if kind == 1:  # keyword and symbol spam
        bits = ["BUY", "$$$", "NOW", "!!!", "click", ">>>", "www.x9.biz", "100%", "FREE", "###"]
        return " ".join(bits[i] for i in rng.integers(0, len(bits), size=30))
    letters = "abcdefghijklmnopqrstuvwxyz"  # lowercase letters in random order
    return " ".join(
        "".join(letters[j] for j in rng.integers(0, 26, size=int(rng.integers(3, 9)))) for _ in range(28)
    )


@check("quality_score gives every planted clean document a higher score than every noisy one")
def test_quality(sub):
    train = has(sub, "train_char_bigram")
    score = has(sub, "quality_score")
    rng = np.random.default_rng(0)
    logp = np.asarray(train([_clean_text(rng, 6) for _ in range(60)]))
    if logp.shape != (128, 128):
        fail(f"expected a (128, 128) table of log probabilities, got shape {logp.shape}")
    close(np.exp(logp).sum(axis=1), np.ones(128), atol=1e-9, label="each row of the table must sum to 1 after exp")
    clean = [score(_clean_text(rng, 4), logp) for _ in range(20)]
    noisy = [score(_noisy_text(rng, k % 3), logp) for k in range(30)]
    if not min(clean) > max(noisy):
        fail(
            f"expected the lowest clean score to beat the highest noisy score; "
            f"clean range [{min(clean):.2f}, {max(clean):.2f}], noisy range [{min(noisy):.2f}, {max(noisy):.2f}]"
        )
    if not all(np.isfinite(clean + noisy)):
        fail("scores must be finite")
