"""Checks for the encoders and cross-attention exercise.

Wrong implementations these checks are built to catch:
- attention_allowed that applies the padding mask to the query axis (rows)
  instead of the key axis (columns). Caught by "padding_mask and attention_allowed".
- cross_attention that zeroes padded weights after the softmax, so the rows no
  longer sum to 1. Caught by the loop comparison in "cross_attention".
- masked_lm_loss averaged over every position instead of the chosen ones.
  Caught by the hand example in "masked_lm_loss" and by the gradient check.
- mask_tokens that replaces every chosen token with the mask id. Caught by the
  80/10/10 proportions in "mask_tokens".
"""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


def ref_softmax(row):
    e = np.exp(row - row.max())
    return e / e.sum()


@check("padding_mask and attention_allowed put the padding on the key axis, with and without the causal rule")
def test_masks(sub):
    padding_mask = has(sub, "padding_mask")
    attention_allowed = has(sub, "attention_allowed")
    pad = padding_mask([3, 2], 4)
    shape(pad, (2, 4), label="padding mask")
    equal(np.asarray(pad).astype(int).tolist(), [[1, 1, 1, 0], [1, 1, 0, 0]], label="padding mask")

    bi = np.asarray(attention_allowed(pad, False))
    shape(bi, (2, 4, 4), label="bidirectional mask")
    want_bi = np.zeros((2, 4, 4), dtype=int)
    for b, n in enumerate([3, 2]):
        want_bi[b, :, :n] = 1
    equal(bi.astype(int).tolist(), want_bi.tolist(), label="bidirectional mask: every query reads every real key")

    ca = np.asarray(attention_allowed(pad, True))
    want_ca = np.zeros((2, 4, 4), dtype=int)
    for b, n in enumerate([3, 2]):
        for i in range(4):
            for j in range(n):
                want_ca[b, i, j] = int(j <= i)
    equal(ca.astype(int).tolist(), want_ca.tolist(), label="causal mask with padding")
    if not ca[0, 3].any():
        fail("a padded query row is all blocked: the padding mask must act on keys (columns), not queries (rows)")


@check("masked_softmax matches a per-row softmax over allowed entries and stays finite for huge scores")
def test_masked_softmax(sub):
    masked_softmax = has(sub, "masked_softmax")
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(3, 5)) * 3
    allowed = rng.random((3, 5)) < 0.6
    allowed[:, 0] = True
    out = np.asarray(masked_softmax(scores, allowed))
    shape(out, (3, 5), label="weights")
    want = np.zeros((3, 5))
    for i in range(3):
        idx = np.where(allowed[i])[0]
        want[i, idx] = ref_softmax(scores[i, idx])
    close(out, want, atol=1e-12, label="against a per-row loop")
    equal(out[~allowed].tolist(), [0.0] * int((~allowed).sum()), label="blocked entries")
    big = np.array([[1000.0, 1001.0, 5000.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            out = np.asarray(masked_softmax(big, np.array([[True, True, False]])))
        except FloatingPointError as err:
            fail(f"masked_softmax overflowed on {big}: {err}. Subtract the row maximum of the allowed scores")
    close(out, np.array([[0.2689414214, 0.7310585786, 0.0]]), atol=1e-9, label="scores 1000 and 1001, 5000 blocked")


@check("cross_attention reproduces the worked example: weights [0.6698, 0.3302], output [6.6976, 3.3024]")
def test_cross_worked(sub):
    cross_attention = has(sub, "cross_attention")
    q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    out, w = cross_attention(q, K, V, np.array([True, True, False]))
    shape(out, (1, 2), label="output")
    shape(w, (1, 3), label="weights")
    close(w, np.array([[0.66976155, 0.33023845, 0.0]]), atol=1e-6, label="weights with key 3 padded")
    close(out, np.array([[6.6976155, 3.3023845]]), atol=1e-5, label="output with key 3 padded")
    out2, w2 = cross_attention(q, K, V, np.array([True, True, True]))
    close(w2, np.array([[0.401112, 0.197776, 0.401112]]), atol=1e-5, label="weights with no padding")


@check("cross_attention equals a loop version, rows sum to 1, and padded encoder positions cannot change the output")
def test_cross_loop(sub):
    cross_attention = has(sub, "cross_attention")
    rng = np.random.default_rng(1)
    Td, Te, d, dv = 4, 6, 5, 3
    q = rng.normal(size=(Td, d))
    K = rng.normal(size=(Te, d))
    V = rng.normal(size=(Te, dv))
    pad = np.array([True, True, True, True, False, False])
    out, w = cross_attention(q, K, V, pad)
    shape(out, (Td, dv), label="output")
    shape(w, (Td, Te), label="weights")

    want_out = np.zeros((Td, dv))
    want_w = np.zeros((Td, Te))
    for i in range(Td):
        real = [j for j in range(Te) if pad[j]]
        s = np.array([q[i] @ K[j] / np.sqrt(d) for j in real])
        p = ref_softmax(s)
        for p_j, j in zip(p, real):
            want_w[i, j] = p_j
            want_out[i] += p_j * V[j]
    close(w, want_w, atol=1e-10, label="weights against the loop")
    close(out, want_out, atol=1e-10, label="output against the loop")
    close(np.asarray(w).sum(axis=-1), np.ones(Td), atol=1e-10, label="each row of weights sums to 1")

    K2, V2 = K.copy(), V.copy()
    K2[~pad] = rng.normal(size=(2, d)) * 50
    V2[~pad] = rng.normal(size=(2, dv)) * 50
    out2, _ = cross_attention(q, K2, V2, pad)
    close(out2, out, atol=1e-10, label="output after the padded keys and values are replaced by large numbers")


@check("mask_tokens chooses 15 percent of positions, and splits them 80, 10, 10 into mask, random, and unchanged")
def test_mask_tokens(sub):
    mask_tokens = has(sub, "mask_tokens")
    vocab, mask_id = 1000, 1000
    ids = np.random.default_rng(5).integers(0, vocab, size=(400, 500))
    corrupted, chosen = mask_tokens(ids, np.random.default_rng(0), mask_id, vocab)
    corrupted = np.asarray(corrupted)
    chosen = np.asarray(chosen)
    shape(corrupted, ids.shape, label="corrupted")
    shape(chosen, ids.shape, label="chosen")
    equal(corrupted[~chosen].tolist() == ids[~chosen].tolist(), True, label="unchosen positions are left unchanged")
    n = chosen.sum()
    rate = n / ids.size
    if abs(rate - 0.15) > 0.003:
        fail(f"expected about 15 percent of positions chosen, got {rate:.4f} ({n} of {ids.size})")
    c_new = corrupted[chosen]
    c_old = ids[chosen]
    frac_mask = float((c_new == mask_id).mean())
    frac_keep = float(((c_new == c_old)).mean())
    frac_rand = 1.0 - frac_mask - frac_keep
    if abs(frac_mask - 0.8) > 0.01:
        fail(f"expected about 0.80 of chosen positions to become the mask id, got {frac_mask:.4f}")
    if abs(frac_rand - 0.1) > 0.01:
        fail(f"expected about 0.10 of chosen positions to get a random id, got {frac_rand:.4f}")
    if abs(frac_keep - 0.1) > 0.01:
        fail(f"expected about 0.10 of chosen positions to stay unchanged, got {frac_keep:.4f}")
    rand_vals = c_new[(c_new != mask_id) & (c_new != c_old)]
    if rand_vals.min() < 0 or rand_vals.max() >= vocab:
        fail(f"random replacements must lie in range({vocab}), got min {rand_vals.min()} and max {rand_vals.max()}")
    again, chosen2 = mask_tokens(ids, np.random.default_rng(0), mask_id, vocab)
    equal(np.asarray(again).tolist() == corrupted.tolist(), True, label="the same seed gives the same corruption")


@check("masked_lm_loss averages over the chosen positions only: 0.6700 on the worked example, not 1.7920")
def test_loss_value(sub):
    masked_lm_loss = has(sub, "masked_lm_loss")
    logits = np.array([[2.0, 0.5, -1.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]])
    targets = np.array([0, 0, 2])
    loss, grad = masked_lm_loss(logits, targets, np.array([True, False, True]))
    close(loss, 0.67, atol=1e-3, label="loss on positions 0 and 2")
    shape(grad, (3, 3), label="gradient")
    loss1, _ = masked_lm_loss(logits, targets, np.array([False, False, True]))
    close(loss1, np.log(3.0), atol=1e-9, label="one chosen position with equal logits is log 3")
    loss0, grad0 = masked_lm_loss(logits, targets, np.array([False, False, False]))
    close(loss0, 0.0, atol=1e-12, label="no chosen position")
    close(grad0, np.zeros((3, 3)), atol=1e-12, label="gradient with no chosen position")
    big = np.array([[1000.0, 0.0, -1000.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            loss_big, _ = masked_lm_loss(big, np.array([0]), np.array([True]))
        except FloatingPointError as err:
            fail(f"masked_lm_loss overflowed on logits {big}: {err}. Subtract the row maximum before exp")
    close(loss_big, 0.0, atol=1e-9, label="a confident correct prediction")


@check("masked_lm_loss gradient matches finite differences and is zero on rows that are not chosen")
def test_loss_grad(sub):
    masked_lm_loss = has(sub, "masked_lm_loss")
    rng = np.random.default_rng(2)
    T, V = 7, 6
    logits = rng.normal(size=(T, V))
    targets = rng.integers(0, V, size=T)
    chosen = np.array([True, False, True, True, False, False, True])
    _, grad = masked_lm_loss(logits, targets, chosen)
    num = finite_difference(lambda z: masked_lm_loss(z, targets, chosen)[0], logits.copy())
    close(grad, num, atol=1e-6, label="analytic gradient against finite differences")
    close(np.asarray(grad)[~chosen], np.zeros((3, V)), atol=1e-12, label="rows that are not chosen")


@check("span_corrupt builds the input and target of the worked example and keeps every token exactly once")
def test_span_corrupt(sub):
    span_corrupt = has(sub, "span_corrupt")
    ids = list(range(10))
    inputs, targets = span_corrupt(ids, [(2, 2), (8, 1)], 100)
    equal(np.asarray(inputs).tolist(), [0, 1, 100, 4, 5, 6, 7, 101, 9], label="inputs")
    equal(np.asarray(targets).tolist(), [100, 2, 3, 101, 8, 102], label="targets")
    inputs, targets = span_corrupt(list(range(20, 30)), [(0, 3)], 500)
    equal(np.asarray(inputs).tolist(), [500, 23, 24, 25, 26, 27, 28, 29], label="a span at the start, inputs")
    equal(np.asarray(targets).tolist(), [500, 20, 21, 22, 501], label="a span at the start, targets")
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 50, size=40).tolist()
    spans = [(3, 4), (12, 2), (20, 6), (39, 1)]
    inputs, targets = span_corrupt(ids, spans, 1000)
    kept = [t for t in np.asarray(inputs).tolist() if t < 1000]
    taken = [t for t in np.asarray(targets).tolist() if t < 1000]
    equal(sorted(kept + taken), sorted(ids), label="tokens in the input plus tokens in the target")


@check("span_corruption_lengths gives 461 input tokens and 104 target tokens for 512 at 15 percent with mean span 3")
def test_lengths(sub):
    lengths = has(sub, "span_corruption_lengths")
    equal(tuple(int(v) for v in lengths(512)), (461, 104, 26), label="T = 512")
    equal(tuple(int(v) for v in lengths(100)), (90, 21, 5), label="T = 100")
    equal(tuple(int(v) for v in lengths(512, 0.5, 4.0)), (320, 321, 64), label="T = 512, density 0.5, mean span 4")
