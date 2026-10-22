"""Checks for the temperature and sampling exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape

LOGITS = np.array([0.5, -2.0, 3.5, 3.4, -1.0])


@check("apply_temperature gives the hand-worked probabilities at three temperatures")
def test_temperature_hand(sub):
    got = has(sub, "apply_temperature")
    close(got(LOGITS, 1.0), [0.0253, 0.0021, 0.5077, 0.4594, 0.0056], atol=6e-5, label="temperature 1")
    close(got(LOGITS, 0.5), [0.0014, 0.0000, 0.5490, 0.4495, 0.0001], atol=6e-5, label="temperature 0.5")
    close(got(LOGITS, 2.0), [0.0952, 0.0273, 0.4267, 0.4059, 0.0450], atol=6e-5, label="temperature 2")


@check("apply_temperature stays finite at 0.001 and tends to uniform at 1000")
def test_temperature_limits(sub):
    got = has(sub, "apply_temperature")
    cold = got(LOGITS, 0.001)
    if not np.all(np.isfinite(cold)):
        fail(f"temperature 0.001 returned non-finite values {cold}; subtract the max before exp")
    close(cold, [0, 0, 1, 0, 0], atol=1e-12, label="temperature 0.001 is greedy")
    hot = got(LOGITS, 1000.0)
    close(hot, np.full(5, 0.2), atol=2e-3, label="temperature 1000 is nearly uniform")
    almost(float(hot.sum()), 1.0, tol=1e-12, label="probabilities sum to 1")


@check("A higher temperature lowers the top probability and raises the tail")
def test_temperature_monotone(sub):
    got = has(sub, "apply_temperature")
    taus = [0.25, 0.5, 1.0, 2.0, 4.0]
    tops = [got(LOGITS, t).max() for t in taus]
    tails = [got(LOGITS, t)[[0, 1, 4]].sum() for t in taus]
    for i in range(len(taus) - 1):
        if not tops[i] > tops[i + 1]:
            fail(f"top probability should fall from temperature {taus[i]} to {taus[i + 1]}, got {tops[i]:.4f} then {tops[i + 1]:.4f}")
        if not tails[i] < tails[i + 1]:
            fail(f"junk mass should rise from temperature {taus[i]} to {taus[i + 1]}, got {tails[i]:.4f} then {tails[i + 1]:.4f}")


@check("filter_probs with top_k keeps the k largest and renormalises")
def test_top_k(sub):
    got = has(sub, "filter_probs")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    close(got(probs, top_k=2), [4 / 7, 3 / 7, 0, 0], atol=1e-12, label="top_k=2")
    # Unsorted input: the two largest are entries 3 and 1.
    close(got(np.array([0.1, 0.35, 0.05, 0.5]), top_k=2), [0, 0.35 / 0.85, 0, 0.5 / 0.85], atol=1e-12, label="unsorted top_k=2")
    unchanged = np.array([0.5, 0.3, 0.2])
    close(got(unchanged, top_k=3), unchanged, atol=1e-12, label="top_k equal to the vocabulary size")


@check("filter_probs with top_p keeps the smallest set that reaches p, crossing token included")
def test_top_p(sub):
    got = has(sub, "filter_probs")
    probs = np.array([0.1, 0.5, 0.05, 0.2, 0.15])
    out = got(probs, top_p=0.7)
    close(out, [0, 0.5 / 0.7, 0, 0.2 / 0.7, 0], atol=1e-12, label="top_p=0.7")
    # The top token alone already exceeds p, and it must still be kept.
    close(got(probs, top_p=0.3), [0, 1, 0, 0, 0], atol=1e-12, label="top_p=0.3 keeps the leader")
    close(got(probs, top_p=0.8), [0, 0.5 / 0.85, 0, 0.2 / 0.85, 0.15 / 0.85], atol=1e-12, label="top_p=0.8")


@check("filter_probs with top_p=1.0 keeps everything and does not index out of range")
def test_top_p_one(sub):
    got = has(sub, "filter_probs")
    # The running total of ten 0.1s ends at 0.9999999999999999, not 1.0.
    probs = np.full(10, 0.1)
    out = got(probs, top_p=1.0)
    close(out, probs, atol=1e-12, label="top_p=1.0")
    original = probs.copy()
    got(probs, top_k=3, top_p=0.5)
    close(probs, original, atol=0, label="the input array is not modified")
    close(got(probs), probs, atol=1e-15, label="no filters")


@check("filter_probs applies top_k first, then top_p on the renormalised result")
def test_both_filters(sub):
    got = has(sub, "filter_probs")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    # Top-3 gives [4/9, 3/9, 2/9]. Its running totals are 0.444, 0.778, so p=0.75
    # needs only two tokens. On the original numbers p=0.75 would need three.
    out = got(probs, top_k=3, top_p=0.75)
    close(out, [4 / 7, 3 / 7, 0, 0], atol=1e-12, label="top_k=3, top_p=0.75")


@check("sample_token at temperature 0 is greedy and returns a plain int")
def test_sample_greedy(sub):
    got = has(sub, "sample_token")
    rng = np.random.default_rng(0)
    out = got(LOGITS, rng, temperature=0)
    equal(out, 2, label="greedy choice")
    if type(out) is not int:
        fail(f"sample_token returned {type(out).__name__}, expected a plain int")
    draw = got(LOGITS, rng, temperature=1.0)
    if type(draw) is not int:
        fail(f"sample_token returned {type(draw).__name__} for a random draw, expected a plain int")


@check("sample_token draws from the filtered distribution and never from the cut tail")
def test_sample_distribution(sub):
    got = has(sub, "sample_token")
    probs = has(sub, "apply_temperature")
    filt = has(sub, "filter_probs")
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
