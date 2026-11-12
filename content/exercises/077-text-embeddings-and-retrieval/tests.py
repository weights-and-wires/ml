"""Checks for the text-embeddings-and-retrieval exercise."""

import math

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite, finite_difference, has, shape

# The pooling example from the lesson: one text with three real tokens, padded to length 4.
# The padding rows hold 9s so that a pooling that reads them gives a visibly wrong answer.
TOKENS = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
PAD = np.array([[9.0, 9.0]])
RIGHT = np.vstack([TOKENS, PAD])[None]
LEFT = np.vstack([PAD, TOKENS])[None]
RIGHT_MASK = np.array([[1, 1, 1, 0]])
LEFT_MASK = np.array([[0, 1, 1, 1]])

# The InfoNCE example from the lesson: two queries, their two positives, two hard negatives.
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
POS = np.array([[0.8, 0.6], [0.6, 0.8]])
HARD = np.array([[0.6, -0.8], [0.8, -0.6]])
DOCS = np.vstack([POS, HARD])

# The ranking example from the lesson: six passages, graded relevance, scores from cosines.
SCORES = np.array([0.95, 0.90, 0.80, 0.60, 0.30, -0.20])
GRADES = np.array([0, 3, 0, 2, 1, 0])


# Plausible wrong implementation caught here: last-token pooling that takes index -1 and
# ignores the mask. It passes on left-padded input and returns a padding row on right-padded
# input, so this check feeds it both.
@check("last_token_pool returns the last real token whether padding sits on the right or the left")
def test_last_token_pool(sub):
    fn = has(sub, "last_token_pool")
    got = fn(RIGHT, RIGHT_MASK)
    shape(got, (1, 2), label="last_token_pool output")
    close(got, np.array([[2.0, 2.0]]), atol=1e-12, label="right-padded: the third token, not the padding")
    close(fn(LEFT, LEFT_MASK), np.array([[2.0, 2.0]]), atol=1e-12, label="left-padded")
    both = np.stack([RIGHT[0], LEFT[0]])
    close(fn(both, np.vstack([RIGHT_MASK, LEFT_MASK])), np.array([[2.0, 2.0], [2.0, 2.0]]), atol=1e-12,
          label="one batch holding both paddings")


# Plausible wrong implementation caught here: a mean over every row, padding included, which
# gives [3, 3.25] on the right-padded input instead of [1, 1.3333].
@check("mean_pool averages only the real tokens: [1, 4/3] on the lesson example")
def test_mean_pool(sub):
    fn = has(sub, "mean_pool")
    want = np.array([[1.0, 4.0 / 3.0]])
    close(fn(RIGHT, RIGHT_MASK), want, atol=1e-12, label="right-padded mean")
    close(fn(LEFT, LEFT_MASK), want, atol=1e-12, label="left-padded mean")


@check("l2_normalize and embed give unit rows: [3, 4] becomes [0.6, 0.8], for both poolings")
def test_embed(sub):
    norm = has(sub, "l2_normalize")
    emb = has(sub, "embed")
    close(norm(np.array([[3.0, 4.0], [0.0, -2.0]])), np.array([[0.6, 0.8], [0.0, -1.0]]), atol=1e-12,
          label="normalised rows")
    got = emb(RIGHT, RIGHT_MASK, "last")
    close(got, np.array([[1.0, 1.0]]) / math.sqrt(2.0), atol=1e-12, label="embed with last-token pooling")
    got = emb(RIGHT, RIGHT_MASK, "mean")
    close(got, np.array([[3.0, 4.0]]) / 5.0, atol=1e-12, label="embed with mean pooling: [1, 4/3] over its length")
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(4, 5, 6))
    mask = np.ones((4, 5), dtype=int)
    mask[1, 3:] = 0
    mask[2, :2] = 0
    for pooling in ("last", "mean"):
        e = emb(hidden, mask, pooling)
        close(np.linalg.norm(e, axis=1), np.ones(4), atol=1e-12, label=f"row lengths with {pooling} pooling")


# Plausible wrong implementation caught here: an InfoNCE whose softmax runs over the positives
# only (in-batch negatives and nothing else) gives 0.1269 below. One that adds each query's own
# hard negative but drops the other query's gives a third value, because h2 scores 0.8 against q1.
@check("info_nce_loss matches the hand example at tau 0.1, and hard negatives raise it")
def test_info_nce_hand(sub):
    fn = has(sub, "info_nce_loss")
    almost(fn(Q, DOCS, 0.1), 0.4735020170871318, 1e-9, label="loss with 2 positives and 2 hard negatives")
    almost(fn(Q, POS, 0.1), 0.12692801104297238, 1e-9, label="loss with in-batch negatives only")
    almost(fn(Q, DOCS, 0.02), 0.3466189891791895, 1e-9, label="loss at tau 0.02")
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 4))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    same = np.vstack([v[0]] * 5)
    almost(fn(np.vstack([v[0]] * 3), same, 0.05), math.log(5.0), 1e-9, label="loss when all 5 candidates are equal, log 5")


@check("info_nce_grad matches finite differences for the queries and for the passages")
def test_info_nce_grad(sub):
    loss = has(sub, "info_nce_loss")
    grad = has(sub, "info_nce_grad")
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 4))
    docs = rng.normal(size=(6, 4))
    tau = 0.5
    dq, ddocs = grad(q, docs, tau)
    shape(dq, q.shape, label="dq")
    shape(ddocs, docs.shape, label="ddocs")
    close(dq, finite_difference(lambda x: loss(x, docs, tau), q), atol=1e-6, rtol=1e-5, label="dq against finite differences")
    close(ddocs, finite_difference(lambda x: loss(q, x, tau), docs), atol=1e-6, rtol=1e-5,
          label="ddocs against finite differences")


@check("recall_at_k counts relevant passages found in the top k: 1/3 at k=3 and 3/3 at k=5 on the lesson corpus")
def test_recall(sub):
    fn = has(sub, "recall_at_k")
    almost(fn(SCORES, GRADES, 3), 1.0 / 3.0, 1e-12, label="recall@3")
    almost(fn(SCORES, GRADES, 5), 1.0, 1e-12, label="recall@5")
    almost(fn(SCORES, GRADES, 1), 0.0, 1e-12, label="recall@1, where the top passage is irrelevant")
    almost(fn(SCORES[::-1].copy(), GRADES[::-1].copy(), 3), 1.0 / 3.0, 1e-12, label="recall@3 on the reversed corpus")
    almost(fn(SCORES, np.zeros(6, dtype=int), 3), 0.0, 1e-12, label="recall@3 when nothing is relevant")


# Plausible wrong implementation caught here: nDCG without the ideal-DCG normaliser, which
# returns the raw DCG 1.8928 for the lesson corpus instead of 0.3975 and exceeds 1 on a
# perfect ranking with large grades.
@check("ndcg_at_k divides by the ideal DCG: 0.3975 at k=3 on the lesson corpus, and 1 for a perfect ranking")
def test_ndcg(sub):
    fn = has(sub, "ndcg_at_k")
    almost(fn(SCORES, GRADES, 3), 0.3974895222916885, 1e-9, label="nDCG@3")
    almost(fn(SCORES, GRADES, 5), 3.1410 / 4.7619, 1e-4, label="nDCG@5")
    almost(fn(GRADES.astype(float), GRADES, 6), 1.0, 1e-12, label="nDCG when the scores equal the grades")
    perfect = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    almost(fn(perfect, np.array([3, 3, 2, 1, 0, 0]), 4), 1.0, 1e-12, label="nDCG of a perfect ranking with grades up to 3")
    almost(fn(SCORES, np.zeros(6, dtype=int), 3), 0.0, 1e-12, label="nDCG when every grade is 0")


def _toy_corpus(seed, d=16, n_topics=4, per_topic=3, max_len=6, pad_value=7.0):
    """Texts whose last real token carries the topic, as a causal model's summary token would."""
    rng = np.random.default_rng(seed)
    topics = np.linalg.qr(rng.normal(size=(d, n_topics)))[0].T  # orthonormal topic directions
    n = n_topics * per_topic
    lengths = rng.integers(3, max_len + 1, size=n)
    labels = np.repeat(np.arange(n_topics), per_topic)
    right = np.full((n, max_len, d), pad_value)
    left = np.full((n, max_len, d), pad_value)
    mask_r = np.zeros((n, max_len), dtype=int)
    mask_l = np.zeros((n, max_len), dtype=int)
    for i in range(n):
        h = 0.3 * rng.normal(size=(lengths[i], d))
        h[-1] += 3.0 * topics[labels[i]]
        right[i, : lengths[i]] = h
        left[i, max_len - lengths[i]:] = h
        mask_r[i, : lengths[i]] = 1
        mask_l[i, max_len - lengths[i]:] = 1
    q = 0.3 * rng.normal(size=(n_topics, 3, d))
    q[:, -1] += 3.0 * topics
    return right, mask_r, left, mask_l, q, labels


@check("a toy corpus ranks the right topic first: recall@3 and nDCG@3 equal 1, with either padding side")
def test_search_corpus(sub):
    embed = has(sub, "embed")
    search = has(sub, "search")
    recall = has(sub, "recall_at_k")
    ndcg = has(sub, "ndcg_at_k")
    for seed in range(3):
        right, mask_r, left, mask_l, q, labels = _toy_corpus(seed)
        q_mask = np.ones(q.shape[:2], dtype=int)
        doc_r = embed(right, mask_r, "last")
        doc_l = embed(left, mask_l, "last")
        close(doc_l, doc_r, atol=1e-12, label=f"seed {seed}: left- and right-padded embeddings")
        q_emb = embed(q, q_mask, "last")
        finite(doc_r, label="document embeddings")
        top = search(q_emb, doc_r, 3)
        shape(top, (4, 3), label="search output")
        for topic in range(4):
            grades = (labels == topic).astype(int) * 2
            scores = q_emb[topic] @ doc_r.T
            got_recall = recall(scores, grades, 3)
            got_ndcg = ndcg(scores, grades, 3)
            if got_recall != 1.0 or abs(got_ndcg - 1.0) > 1e-12:
                fail(f"seed {seed}, query {topic}: expected recall@3 = 1 and nDCG@3 = 1, "
                     f"got {got_recall:.3f} and {got_ndcg:.3f}; top 3 = {top[topic].tolist()}")
            equal(sorted(top[topic].tolist()), sorted(np.flatnonzero(labels == topic).tolist()),
                  label=f"seed {seed}, query {topic}: the three passages search returns")
