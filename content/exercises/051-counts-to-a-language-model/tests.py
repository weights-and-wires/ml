"""Checks for the count model, the embedding gradient, and the toy vectors.

Two plausible mistakes are named here and must fail:

- add_k_observed_only adds k only where the count is already positive, so an
  unseen next word stays at probability 0.
- embedding_backward_overwrite writes each gradient row into the table and
  keeps only the last write when the same index appears twice.
"""

import math

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has


PREV = np.array([0, 1, 2, 0, 1, 3])
NXT = np.array([1, 2, 4, 1, 3, 4])
VOCAB = 5

# start big small dog hound cat kitten runs sits
LINES = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 4, 7],
        [0, 2, 5, 8],
        [0, 2, 6, 8],
    ],
    dtype=int,
)
NOTEBOOK = np.repeat(LINES, 4, axis=0)


def reference_counts() -> np.ndarray:
    counts = np.zeros((VOCAB, VOCAB), dtype=float)
    np.add.at(counts, (PREV, NXT), 1.0)
    return counts


def reference_probs(counts: np.ndarray, k: float) -> np.ndarray:
    smoothed = counts + k
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def add_k_observed_only(counts: np.ndarray, k: float) -> np.ndarray:
    """The wrong smoother: k is added only on cells that are already positive."""
    bumped = np.array(counts, dtype=float)
    bumped[counts > 0] += k
    row_sum = bumped.sum(axis=1, keepdims=True)
    return np.divide(bumped, row_sum, out=np.zeros_like(bumped), where=row_sum > 0)


def embedding_backward_overwrite(
    grad_out: np.ndarray, indices: np.ndarray, vocab_size: int
) -> np.ndarray:
    """The wrong backward: a repeated index keeps only the last gradient row."""
    gradient = np.zeros((vocab_size, grad_out.shape[1]), dtype=float)
    gradient[np.asarray(indices, dtype=int)] = grad_out
    return gradient


@check("bigram_counts matches the six-pair tally: start to big is 2, and big to dog is 1")
def test_bigram_counts(sub):
    fn = has(sub, "bigram_counts")
    got = fn(PREV, NXT, VOCAB)
    want = reference_counts()
    close(got, want, atol=0, rtol=0, label="bigram counts")


@check("add-1 probabilities match the hand fractions, including 1/7 for an unseen word")
def test_add_k(sub):
    # add_k_observed_only leaves P(dog | start) at 0. The smoothed table requires 1/7.
    fn = has(sub, "add_k_probs")
    counts = reference_counts()
    got = np.asarray(fn(counts, 1.0), dtype=float)
    want = reference_probs(counts, 1.0)
    wrong = add_k_observed_only(counts, 1.0)
    if got.shape == want.shape and np.allclose(got, wrong) and not np.allclose(wrong, want):
        fail(
            f"P(dog | start) is {got[0, 2]:.6g}, expected {want[0, 2]:.6g}. "
            "Adding k only to cells that were already positive leaves an unseen word at 0."
        )
    close(got, want, atol=1e-8, rtol=0, label="add-1 probabilities")
    if not np.allclose(got[4], np.full(VOCAB, 0.2)):
        fail(
            f"the empty row for runs is {got[4]}, expected a uniform row of 0.2"
        )


@check("the six pairs have mean negative log likelihood 1.0662")
def test_mean_nll(sub):
    fn = has(sub, "mean_nll")
    probs = reference_probs(reference_counts(), 1.0)
    got = float(fn(probs, PREV, NXT))
    want = float(-np.mean(np.log(probs[PREV, NXT])))
    almost(got, want, tol=1e-8, label="mean nll of the six pairs")
    uniform = np.full((VOCAB, VOCAB), 1.0 / VOCAB)
    got_uniform = float(fn(uniform, PREV, NXT))
    almost(got_uniform, math.log(VOCAB), tol=1e-8, label="mean nll of a uniform table")


@check("perplexity of the tally is 2.904, and perplexity of a uniform table is 5")
def test_perplexity(sub):
    nll_fn = has(sub, "mean_nll")
    ppl_fn = has(sub, "perplexity")
    probs = reference_probs(reference_counts(), 1.0)
    nll = float(nll_fn(probs, PREV, NXT))
    got = float(ppl_fn(nll))
    want = math.exp(nll)
    almost(got, want, tol=1e-8, label="perplexity of the six pairs")
    uniform = np.full((VOCAB, VOCAB), 1.0 / VOCAB)
    uniform_nll = float(nll_fn(uniform, PREV, NXT))
    got_uniform = float(ppl_fn(uniform_nll))
    almost(got_uniform, float(VOCAB), tol=1e-6, label="perplexity of a uniform distribution")


@check("softmax of ln(count + 1) reproduces add-k, and a row of 1000s stays uniform")
def test_row_softmax(sub):
    fn = has(sub, "row_softmax")
    counts = reference_counts()
    got = fn(np.log(counts + 1.0))
    want = reference_probs(counts, 1.0)
    close(got, want, atol=1e-8, rtol=0, label="softmax of log counts")
    flat = fn(np.full((1, 3), 1000.0))
    close(flat, np.full((1, 3), 1.0 / 3.0), atol=1e-6, rtol=0, label="softmax of a row of 1000s")


@check("embedding_forward copies rows, and a repeated index copies the row twice")
def test_embedding_forward(sub):
    fn = has(sub, "embedding_forward")
    table = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    indices = np.array([1, 0, 1])
    got = fn(table, indices)
    want = np.array([[0.0, 2.0], [1.0, 0.0], [0.0, 2.0]])
    close(got, want, atol=0, rtol=0, label="lookup of indices [1, 0, 1]")


@check("embedding backward adds a repeated row to (1, 3), and matches a finite difference")
def test_embedding_backward(sub):
    # embedding_backward_overwrite would leave row 1 at (0, 3), the last write.
    backward = has(sub, "embedding_backward")
    forward = has(sub, "embedding_forward")
    grad_out = np.array([[1.0, 0.0], [4.0, 1.0], [0.0, 3.0]])
    indices = np.array([1, 0, 1])
    got = np.asarray(backward(grad_out, indices, 3), dtype=float)
    want = np.array([[4.0, 1.0], [1.0, 3.0], [0.0, 0.0]])
    wrong = embedding_backward_overwrite(grad_out, indices, 3)
    if got.shape == want.shape and np.allclose(got, wrong) and not np.allclose(wrong, want):
        fail(
            f"row 1 is {got[1]}, expected {want[1]}. "
            "An overwrite keeps the last gradient (0, 3) and drops the first."
        )
    close(got, want, atol=1e-8, rtol=0, label="accumulated embedding gradient")

    rng = np.random.default_rng(0)
    table = rng.normal(size=(6, 4))
    sample = np.array([1, 4, 1, 2])
    weights = rng.normal(size=(4, 4))

    def loss_of(flat: np.ndarray) -> float:
        rows = forward(flat.reshape(6, 4), sample)
        return float(np.sum(rows * weights))

    numeric = finite_difference(loss_of, table.ravel(), eps=1e-6).reshape(6, 4)
    analytic = backward(weights, sample, 6)
    close(analytic, numeric, atol=1e-5, rtol=1e-5, label="embedding gradient versus finite difference")


@check("skip-gram loss is 1.0064 on the hand vectors, and the gradient matches a finite difference")
def test_skipgram_grad(sub):
    fn = has(sub, "skipgram_loss_and_grad")
    center = np.array([1.0, 0.0])
    positive = np.array([1.0, 0.0])
    negatives = np.array([[0.0, 1.0]])
    loss, grad_center, grad_positive, grad_negatives = fn(center, positive, negatives)
    almost(float(loss), 1.0064088680781682, tol=1e-6, label="skip-gram loss")
    close(grad_center, np.array([-0.2689414213699951, 0.5]), atol=1e-6, rtol=0, label="center gradient")
    close(grad_positive, np.array([-0.2689414213699951, 0.0]), atol=1e-6, rtol=0, label="positive gradient")
    close(grad_negatives, np.array([[0.5, 0.0]]), atol=1e-6, rtol=0, label="noise gradient")

    rng = np.random.default_rng(1)
    width, noise_count = 3, 2
    center = rng.normal(size=width)
    positive = rng.normal(size=width)
    negatives = rng.normal(size=(noise_count, width))
    packed = np.concatenate([center, positive, negatives.ravel()])

    def loss_of(vec: np.ndarray) -> float:
        value, *_rest = fn(
            vec[:width],
            vec[width : 2 * width],
            vec[2 * width :].reshape(noise_count, width),
        )
        return float(value)

    numeric = finite_difference(loss_of, packed.copy(), eps=1e-6)
    _loss, grad_center, grad_positive, grad_negatives = fn(center, positive, negatives)
    analytic = np.concatenate(
        [np.asarray(grad_center).ravel(), np.asarray(grad_positive).ravel(), np.asarray(grad_negatives).ravel()]
    )
    close(analytic, numeric, atol=1e-5, rtol=1e-5, label="skip-gram gradient versus finite difference")


@check("the window model prefers runs over sits after big dog, and beats the uniform loss")
def test_window_lm(sub):
    fn = has(sub, "train_window_lm")
    nll, probs = fn(
        NOTEBOOK, 9, window=2, dim=8, hidden=16, steps=80, lr=0.5, seed=0,
        query_context=np.array([1, 3]),
    )
    nll = float(nll)
    probs = np.asarray(probs, dtype=float)
    uniform = math.log(9)
    if nll >= uniform - 0.8:
        fail(
            f"final mean NLL is {nll:.4f}, expected below {uniform - 0.8:.4f} "
            f"(a uniform model scores {uniform:.4f})"
        )
    if probs.shape != (9,):
        fail(f"query probabilities have shape {probs.shape}, expected (9,)")
    total = float(probs.sum())
    if abs(total - 1.0) > 1e-5:
        fail(f"query probabilities sum to {total:.6f}, expected 1")
    runs = float(probs[7])
    sits = float(probs[8])
    if runs <= sits + 0.5:
        fail(
            f"P(runs | big, dog) is {runs:.4f} and P(sits | big, dog) is {sits:.4f}, "
            "expected runs to lead by more than 0.5"
        )


@check("after skip-gram, dog is nearer hound than sits by more than 0.25 cosine")
def test_skipgram_neighbour(sub):
    fn = has(sub, "train_skipgram")
    table = np.asarray(
        fn(NOTEBOOK, 9, window=1, dim=8, epochs=20, lr=0.4, n_negative=4, seed=0),
        dtype=float,
    )
    if table.shape != (9, 8):
        fail(f"center table has shape {table.shape}, expected (9, 8)")

    def cosine(left: int, right: int) -> float:
        a, b = table[left], table[right]
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    hound = cosine(3, 4)
    sits = cosine(3, 8)
    if hound <= sits + 0.25:
        fail(
            f"cosine(dog, hound) is {hound:.4f} and cosine(dog, sits) is {sits:.4f}, "
            "expected hound to lead by more than 0.25"
        )
