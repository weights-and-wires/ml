"""Checks for the byte-pair encoding exercise: training, encoding, and decoding."""

import numpy as np

from mlcheck import check, equal, fail, has

CLASSIC = [(97, 97), (97, 98), (256, 257)]
ALPHABET = list("the cat sat on a mat, éè 世界 🎉\n\t.")


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


def test_counts_runs(sub):
    f = has(sub, "get_pair_counts")
    got = dict(f([[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]]))
    want = {(97, 97): 4, (97, 98): 2, (98, 100): 1, (100, 97): 1, (98, 97): 1, (97, 99): 1}
    if got != want:
        fail(f"counts on aaabdaaabac were {got}, expected {want}")


@check("get_pair_counts counts pairs inside each word, never across words, and counts overlapping runs by position")
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
    test_counts_runs(sub)


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


@check("merge_pair glues non-overlapping pairs from the left, keeps its input intact, and never changes what the tokens expand to")
def test_merge(sub):
    f = has(sub, "merge_pair")
    same(f([[1, 2, 3, 1, 2], [1, 2]], (1, 2), 9), [[9, 3, 9], [9]], "merge of (1, 2)")
    same(f([[5, 5, 5]], (5, 5), 20), [[20, 5]], "merge of a run of three")
    same(f([[5, 5, 5, 5]], (5, 5), 20), [[20, 20]], "merge of a run of four")
    test_merge_property(sub)


@check("train_bpe on aaabdaaabac learns the three merges worked by hand")
def test_classic(sub):
    f = has(sub, "train_bpe")
    merges, vocab = f("aaabdaaabac", 259)
    equal(merges, [(97, 97), (97, 98), (256, 257)], label="merges")
    equal(vocab[256], b"aa", label="vocab[256]")
    equal(vocab[257], b"ab", label="vocab[257]")
    equal(vocab[258], b"aaab", label="vocab[258]")
    equal(vocab[97], b"a", label="vocab[97]")


def test_boundary(sub):
    f = has(sub, "train_bpe")
    merges, vocab = f("a a a a a", 300)
    equal(merges, [(32, 97)], label="merges for five single-letter words")
    equal(vocab[256], b" a", label="vocab[256]")


@check("train_bpe breaks a tie toward the smaller pair and never merges across a word boundary")
def test_tie(sub):
    f = has(sub, "train_bpe")
    merges, _ = f("ab\ncd", 257)
    equal(merges, [(97, 98)], label="first merge when (97, 98) and (99, 100) tie")
    merges, _ = f("cd\nab", 257)
    equal(merges, [(97, 98)], label="first merge does not depend on order")
    test_boundary(sub)


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


def test_word_edges(sub):
    f = has(sub, "apply_merges_to_word")
    word = [97, 97, 97, 98]
    f(word, CLASSIC)
    equal(word, [97, 97, 97, 98], label="the input list")
    got = list(f([97, 98], [(256, 257), (97, 98)]))
    equal(got, [257], label="a merge on ids that do not exist is idle")

def test_order(sub):
    f = has(sub, "apply_merges_to_word")
    word = list(b"aaabdaaabac")
    equal(list(f(word, list(reversed(CLASSIC)))), [258, 257, 100, 258, 257, 97, 99], label="reversed order")
    if len(f(word, list(reversed(CLASSIC)))) <= len(f(word, CLASSIC)):
        fail("the reversed order should give more tokens than the learned order")


@check("apply_merges_to_word replays the merges in order, and the reversed order gives 7 tokens, not 5")
def test_word(sub):
    f = has(sub, "apply_merges_to_word")
    equal(list(f([97, 97, 97, 98], CLASSIC)), [258], label="aaab")
    got = list(f(list(b"aaabdaaabac"), CLASSIC))
    equal(got, [258, 100, 258, 97, 99], label="aaabdaaabac")
    equal(list(f([104, 105], [])), [104, 105], label="no merges")
    test_word_edges(sub)
    test_order(sub)


def test_encode_pieces(sub):
    f = has(sub, "encode")
    train = has(sub, "train_bpe")
    equal(list(f("aaabdaaabac", CLASSIC)), [258, 100, 258, 97, 99], label="classic")
    merges, _ = train("the cat sat on the mat. the dog sat on the log. " * 4, 290)
    whole = list(f("the cat sat", merges))
    parts = list(f("the", merges)) + list(f(" cat", merges)) + list(f(" sat", merges))
    equal(whole, parts, label="encode of the sentence against its pieces")
    again = list(f("a cat sat", merges))
    equal(again[-len(list(f(" cat sat", merges))):], list(f(" cat sat", merges)),
          label="a word gets the same ids whatever comes before it")


@check("encode gives raw bytes with no merges and replays the merges inside each piece, never across pieces")
def test_encode_raw(sub):
    f = has(sub, "encode")
    for text in ("hello, world", "héllo 世界 🎉", "  two  spaces\tand a tab\n", ""):
        equal(list(f(text, [])), list(text.encode("utf-8")), label=f"bytes of {text!r}")
    test_encode_pieces(sub)


@check("decode joins the bytes first, so it handles a character split across two tokens")
def test_decode(sub):
    f = has(sub, "decode")
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"\xf0\x9f"
    vocab[257] = b"\x8e\x89"
    equal(f([256, 257], vocab), "🎉", label="emoji cut after its second byte")
    equal(f([], vocab), "", label="empty ids")
    equal(f([72, 105], vocab), "Hi", label="plain bytes")


@check("decode(encode(text)) is exactly text after training, with accents, Chinese, and emoji")
def test_round_trip(sub):
    enc = has(sub, "encode")
    dec = has(sub, "decode")
    train = has(sub, "train_bpe")
    corpus = "the cat sat on the mat. the dog sat on the log. é " * 5
    merges, vocab = train(corpus, 300)
    rng = np.random.default_rng(0)
    texts = ["héllo — 世界 🎉", "the cat  sat\non the mat.", ""]
    for _ in range(20):
        texts.append("".join(rng.choice(ALPHABET, size=int(rng.integers(1, 40)))))
    for text in texts:
        ids = enc(text, merges)
        equal(dec(ids, vocab), text, label=f"round trip of {text!r}")
        if any(i >= 256 + len(merges) for i in ids):
            fail(f"an id is outside the vocabulary for {text!r}: {ids}")
    short = enc("the cat sat on the mat.", merges)
    if len(short) >= len("the cat sat on the mat."):
        fail(f"trained merges should shorten text it was trained on, got {len(short)} ids")
