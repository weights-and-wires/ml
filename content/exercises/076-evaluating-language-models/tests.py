"""Checks for the evaluating-language-models exercise."""

import math

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


@check("bits_per_byte: a model that is uniform over 256 byte values scores 8 bits per byte")
def test_bpb_uniform(sub):
    fn = has(sub, "bits_per_byte")
    n_bytes = 50
    got = fn(n_bytes * math.log(256), n_bytes)
    almost(got, 8.0, 1e-9, label="uniform bytes, expected 8.0 bits per byte")


@check("bits_per_byte: two tokenisations of one text with the same total loss give the same value")
def test_bpb_tokenisers(sub):
    # Wrong implementation caught here and in the previous check: dividing by
    # the number of tokens instead of the number of bytes.
    fn = has(sub, "bits_per_byte")
    n_bytes = 40
    total = n_bytes * math.log(2)  # exactly 1 bit per byte
    tok_a = [total / 10] * 10  # 10 tokens, 2.773 nats each
    tok_b = [total / 20] * 20  # 20 tokens, 1.386 nats each
    a = fn(sum(tok_a), n_bytes)
    b = fn(sum(tok_b), n_bytes)
    almost(a, 1.0, 1e-9, label="tokeniser A, expected 1.0 bpb")
    almost(b, 1.0, 1e-9, label="tokeniser B, expected 1.0 bpb")


@check("choose_answer: raw and length-normalised scores pick different options on the worked example")
def test_choose(sub):
    fn = has(sub, "choose_answer")
    lp = np.array([-6.0, -9.0, -12.0])
    ln = np.array([3, 6, 12])
    raw = fn(lp, ln, False)
    norm = fn(lp, ln, True)
    if raw != 0:
        fail(f"raw scores [-6, -9, -12] should pick option 0, got {raw}")
    if norm != 2:
        fail(f"per-byte scores [-2, -1.5, -1] should pick option 2, got {norm}")
    if not isinstance(raw, int):
        fail(f"expected a Python int, got {type(raw).__name__}")
    equal(fn(np.array([-2.0, -2.0]), np.array([1, 1]), True), 0, label="a tie goes to the first option")


@check("pass_at_k: hand values, including the with-replacement trap")
def test_pass_hand(sub):
    # Wrong implementation caught here: 1 - (1 - c / n) ** k, which samples
    # with replacement. It gives 0.64 for (5, 2, 2) where the answer is 0.7.
    fn = has(sub, "pass_at_k")
    almost(fn(5, 2, 2), 0.7, 1e-12, label="pass_at_k(5, 2, 2), expected 0.7 (not 0.64)")
    almost(fn(10, 0, 3), 0.0, 1e-12, label="no passing sample")
    almost(fn(10, 4, 1), 0.4, 1e-12, label="k = 1 equals c / n")
    almost(fn(6, 5, 2), 1.0, 1e-12, label="n - c < k forces a pass")
    almost(fn(8, 8, 8), 1.0, 1e-12, label="all correct")


@check("pass_at_k stays finite and exact for n = 5000, where binomials overflow")
def test_pass_stable(sub):
    fn = has(sub, "pass_at_k")
    n, c, k = 5000, 50, 2000
    want = 1.0 - math.comb(n - c, k) / math.comb(n, k)  # exact integer ratio
    got = fn(n, c, k)
    if not np.isfinite(got):
        fail(f"pass_at_k(5000, 50, 2000) returned {got}, expected {want:.12f}")
    almost(got, want, 1e-9, label="pass_at_k(5000, 50, 2000)")


@check("pass_at_k matches a Monte Carlo draw of k samples without replacement")
def test_pass_mc(sub):
    fn = has(sub, "pass_at_k")
    rng = np.random.default_rng(0)
    n, c, k, trials = 20, 6, 5, 40000
    order = np.argsort(rng.uniform(size=(trials, n)), axis=1)[:, :k]
    mc = float(np.mean((order < c).any(axis=1)))  # samples 0..c-1 are the correct ones
    got = fn(n, c, k)
    almost(got, mc, 0.01, label=f"pass_at_k(20, 6, 5) against Monte Carlo {mc:.4f}")
    naive = 1 - (1 - c / n) ** k
    if abs(got - naive) < 0.01:
        fail(f"pass_at_k {got:.4f} equals the with-replacement value {naive:.4f}; it should differ")


@check("bootstrap_ci covers the true mean about 95 percent of the time over seeded repeats")
def test_bootstrap_coverage(sub):
    # Wrong implementation caught here: taking percentiles of the raw scores
    # instead of the resampled means. The interval is then [0, 1] and covers
    # 100 percent of the time, with a width that never shrinks.
    fn = has(sub, "bootstrap_ci")
    rng = np.random.default_rng(1)
    true_mean, n, repeats = 0.7, 100, 300
    covered, widths = 0, []
    for _ in range(repeats):
        scores = (rng.uniform(size=n) < true_mean).astype(float)
        lo, hi = fn(scores, 300, rng)
        covered += lo <= true_mean <= hi
        widths.append(hi - lo)
    rate = covered / repeats
    if not 0.90 <= rate <= 0.99:
        fail(f"coverage over {repeats} repeats was {rate:.3f}, expected between 0.90 and 0.99")
    w = float(np.mean(widths))
    # The standard error is sqrt(0.7 * 0.3 / 100) = 0.0458, so a 95 percent width is about 0.18.
    almost(w, 0.18, 0.03, label="mean interval width, expected about 0.18")


@check("bootstrap_ci is reproducible for a seed and brackets the sample mean")
def test_bootstrap_basic(sub):
    fn = has(sub, "bootstrap_ci")
    scores = np.array([1.0, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 5)
    a = fn(scores, 500, np.random.default_rng(3))
    b = fn(scores, 500, np.random.default_rng(3))
    equal(tuple(a), tuple(b), label="same seed, same interval")
    if not a[0] < scores.mean() < a[1]:
        fail(f"interval {a} should contain the sample mean {scores.mean():.3f}")


@check("paired_bootstrap_diff: pairing makes the gap significant where separate resampling does not")
def test_paired(sub):
    # Wrong implementation caught here: drawing two index arrays, one per
    # model. That discards the pairing, and the interval grows to about
    # (-0.14, 0.05), which contains 0.
    fn = has(sub, "paired_bootstrap_diff")
    a = np.zeros(200)
    a[:120] = 1.0
    b = a.copy()
    b[:12] = 0.0  # A right, B wrong on 12 items
    b[120:123] = 1.0  # A wrong, B right on 3 items
    diff, lo, hi = fn(a, b, 2000, np.random.default_rng(0))
    almost(diff, -0.045, 1e-12, label="observed difference, expected -0.045")
    if not (lo < diff < hi):
        fail(f"interval ({lo:.4f}, {hi:.4f}) should contain the observed difference {diff:.4f}")
    if hi >= 0:
        fail(f"paired interval ({lo:.4f}, {hi:.4f}) should lie below 0; the models disagree on only 15 items")
    if hi - lo > 0.09:
        fail(f"paired interval width {hi - lo:.4f} is too wide; expected about 0.075")


@check("ngram_overlap: hand example, short documents, and planted contaminated items")
def test_overlap(sub):
    fn = has(sub, "ngram_overlap")
    almost(fn([list("xbcdy")], list("abcde"), 3), 1 / 3, 1e-12, label="a b c d e against x b c d y, n = 3")
    almost(fn([list("abcde")], list("ab"), 3), 0.0, 1e-12, label="test document shorter than n")
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(400)]
    train = [list(rng.choice(vocab, size=60)) for _ in range(10)]
    planted = [list(train[i][5:45]) for i in range(5)]  # verbatim 40-token excerpts
    for doc in planted:
        doc[20] = "edit"  # one edited token breaks 8 of the 8-grams, not all of them
    clean = [list(rng.choice(vocab, size=40)) for _ in range(5)]
    hot = [fn(train, d, 8) for d in planted]
    cold = [fn(train, d, 8) for d in clean]
    if min(hot) < 0.7:
        fail(f"planted items should overlap above 0.7, got {np.round(hot, 3).tolist()}")
    if max(cold) > 0.05:
        fail(f"clean items should overlap below 0.05, got {np.round(cold, 3).tolist()}")
