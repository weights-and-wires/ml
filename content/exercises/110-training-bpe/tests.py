"""Checks for the BPE training exercise."""

import numpy as np

from mlcheck import check, equal, fail, has


def same(got, want, label):
    """Compare nested lists without turning them into arrays."""
    got = [list(w) for w in got]
    if got != want:
        fail(f"{label}: got {got}, expected {want}")


def expand(ids, table):
    """Expand merged ids back to the ids they were built from."""
    out = []
    for i in ids:
        out.extend(expand(table[i], table) if i in table else [i])
    return out


@check("get_pair_counts counts adjacent pairs inside each word, never across words")
def test_counts(sub):
    f = has(sub, "get_pair_counts")
    got = dict(f([[1, 2, 3, 1, 2], [1, 2]]))
    want = {(1, 2): 3, (2, 3): 1, (3, 1): 1}
    if got != want:
        fail(f"get_pair_counts returned {got}, expected {want}")
    if dict(f([[7]])) != {}:
        fail("a single id has no neighbour, so the counts should be empty")
    got = dict(f([[1], [2]]))
    if got != {}:
        fail(f"the end of one word and the start of the next are not a pair, got {got}")


@check("get_pair_counts counts overlapping runs by position")
def test_counts_runs(sub):
    f = has(sub, "get_pair_counts")
    got = dict(f([[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]]))
    want = {(97, 97): 4, (97, 98): 2, (98, 100): 1, (100, 97): 1, (98, 97): 1, (97, 99): 1}
    if got != want:
        fail(f"counts on aaabdaaabac were {got}, expected {want}")


@check("merge_pair glues non-overlapping pairs from the left, so [5, 5, 5] gives [20, 5]")
def test_merge(sub):
    f = has(sub, "merge_pair")
    same(f([[1, 2, 3, 1, 2], [1, 2]], (1, 2), 9), [[9, 3, 9], [9]], "merge of (1, 2)")
    same(f([[5, 5, 5]], (5, 5), 20), [[20, 5]], "merge of a run of three")
    same(f([[5, 5, 5, 5]], (5, 5), 20), [[20, 20]], "merge of a run of four")


@check("merge_pair keeps its input intact and never changes what the tokens expand to")
def test_merge_property(sub):
    f = has(sub, "merge_pair")
    rng = np.random.default_rng(0)
    for _ in range(30):
        words = [list(map(int, rng.integers(1, 4, size=int(rng.integers(1, 12))))) for _ in range(4)]
        before = [list(w) for w in words]
        out = [list(w) for w in f(words, (1, 2), 50)]
        same(words, before, "the input lists after merge_pair")
        for old, new in zip(before, out):
            if expand(new, {50: [1, 2]}) != old:
                fail(f"{new} expands to {expand(new, {50: [1, 2]})}, expected {old}")
            if 50 not in new and (1, 2) in list(zip(old, old[1:])):
                fail(f"{old} holds the pair (1, 2) but {new} still has no merged id")


@check("train_bpe on aaabdaaabac learns the three merges worked by hand")
def test_classic(sub):
    f = has(sub, "train_bpe")
    merges, vocab = f("aaabdaaabac", 259)
    equal(merges, [(97, 97), (97, 98), (256, 257)], label="merges")
    equal(vocab[256], b"aa", label="vocab[256]")
    equal(vocab[257], b"ab", label="vocab[257]")
    equal(vocab[258], b"aaab", label="vocab[258]")
    equal(vocab[97], b"a", label="vocab[97]")


@check("train_bpe breaks a tie toward the smaller pair")
def test_tie(sub):
    f = has(sub, "train_bpe")
    merges, _ = f("ab\ncd", 257)
    equal(merges, [(97, 98)], label="first merge when (97, 98) and (99, 100) tie")
    merges, _ = f("cd\nab", 257)
    equal(merges, [(97, 98)], label="first merge does not depend on order")


@check("train_bpe never merges across a word boundary")
def test_boundary(sub):
    f = has(sub, "train_bpe")
    merges, vocab = f("a a a a a", 300)
    equal(merges, [(32, 97)], label="merges for five single-letter words")
    equal(vocab[256], b" a", label="vocab[256]")


@check("train_bpe reaches the requested size and stops early when no pair is left")
def test_size(sub):
    f = has(sub, "train_bpe")
    corpus = ("the quick brown fox jumps over the lazy dog while the sleepy cat "
              "watches from a warm windowsill in the afternoon sun ") * 10
    merges, vocab = f(corpus, 280)
    equal(len(vocab), 280, label="vocabulary size")
    equal(len(merges), 24, label="merge count")
    merges, vocab = f("ab", 400)
    equal(merges, [(97, 98)], label="merges for a two-byte corpus")
    equal(len(vocab), 257, label="vocabulary size when no pair remains")
