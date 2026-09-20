"""Checks for the byte-pair encoding exercise."""

from mlcheck import check, equal, fail, has


@check("get_pair_counts counts every adjacent pair across every word")
def test_get_pair_counts(sub):
    got = has(sub, "get_pair_counts")
    counts = dict(got([[1, 2, 3, 1, 2], [1, 2]]))
    want = {(1, 2): 3, (2, 3): 1, (3, 1): 1}
    if counts != want:
        fail(f"get_pair_counts returned {counts}, expected {want}")

    empty = dict(got([[7]]))
    if empty != {}:
        fail(f"a single symbol has no adjacent pair, but got {empty}")


@check("merge_pair replaces every non-overlapping occurrence, left to right")
def test_merge_pair(sub):
    got = has(sub, "merge_pair")
    result = [list(word) for word in got([[1, 2, 3, 1, 2], [1, 2]], (1, 2), 9)]
    if result != [[9, 3, 9], [9]]:
        fail(f"merge_pair result is {result}, expected [[9, 3, 9], [9]]")

    # Three identical symbols in a row: merging the pair with itself merges
    # the first two and leaves the third alone, never double-counting.
    triple = [list(word) for word in got([[5, 5, 5]], (5, 5), 20)]
    if triple != [[20, 5]]:
        fail(f"merging an overlapping run gave {triple}, expected [[20, 5]]")


@check("encode and decode round-trip text holding non-ASCII bytes, with no merges")
def test_roundtrip_no_merges(sub):
    encode = has(sub, "encode")
    decode = has(sub, "decode")
    base_vocab = {i: bytes([i]) for i in range(256)}

    for text in ("hello, world", "héllo — 世界 🎉", "  two  spaces\tand a tab\n"):
        ids = encode(text, [])
        if any(not (0 <= i < 256) for i in ids):
            fail(f"with no merges, every ID should be a raw byte 0-255, got {ids}")
        equal(decode(ids, base_vocab), text, label=f"round trip of {text!r}")


@check("train_bpe on a classic short example reproduces the three merges by hand")
def test_train_bpe_classic(sub):
    got = has(sub, "train_bpe")
    merges, vocab = got("aaabdaaabac", 259)

    equal(
        merges,
        [(97, 97), (97, 98), (256, 257)],
        label="merge order on 'aaabdaaabac'",
    )
    equal(vocab[256], b"aa", label="vocab[256]")
    equal(vocab[257], b"ab", label="vocab[257]")
    equal(vocab[258], b"aaab", label="vocab[258]")


@check("train_bpe returns a vocabulary of exactly the requested size")
def test_vocab_size(sub):
    got = has(sub, "train_bpe")
    # A richer, more varied corpus than the exercise's other tests need, so
    # there are enough distinct pairs left to reach a larger target size.
    corpus = (
        "the quick brown fox jumps over the lazy dog while the sleepy cat "
        "watches from a warm windowsill in the afternoon sun, dreaming of "
        "fish and cream and long naps beside a crackling fireplace during "
        "the cold winter evenings "
    ) * 30
    for vocab_size in (260, 300):
        merges, vocab = got(corpus, vocab_size)
        equal(len(vocab), vocab_size, label=f"vocabulary size for target {vocab_size}")
        equal(len(merges), vocab_size - 256, label=f"merge count for target {vocab_size}")


@check("Applying the same merges out of order gives a different, worse tokenisation")
def test_order_matters(sub):
    encode = has(sub, "encode")
    train_bpe = has(sub, "train_bpe")
    merges, _ = train_bpe("aaabdaaabac", 259)

    correct = encode("aaabdaaabac", merges)
    equal(correct, [258, 100, 258, 97, 99], label="correctly ordered encoding")

    reversed_merges = list(reversed(merges))
    scrambled = encode("aaabdaaabac", reversed_merges)
    equal(scrambled, [258, 257, 100, 258, 257, 97, 99], label="reverse-ordered encoding")

    if len(scrambled) <= len(correct):
        fail(
            "applying the merges out of order should produce more tokens, not "
            f"fewer or the same: got {len(scrambled)} against {len(correct)} "
            "for the correct order"
        )


@check("encode and decode round-trip text a trained tokeniser never saw, including non-ASCII bytes")
def test_roundtrip_after_training(sub):
    encode = has(sub, "encode")
    decode = has(sub, "decode")
    train_bpe = has(sub, "train_bpe")

    corpus = "the quick brown fox jumps over the lazy dog. " * 30
    merges, vocab = train_bpe(corpus, 280)

    for text in ("the quick fox", "héllo — 世界 🎉", "some completely unseen! text?"):
        ids = encode(text, merges)
        equal(decode(ids, vocab), text, label=f"round trip of {text!r} after training")


@check("Fertility falls as the trained vocabulary grows")
def test_fertility_falls(sub):
    train_bpe = has(sub, "train_bpe")
    fertility = has(sub, "fertility")

    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox runs from the lazy dog. "
    ) * 20
    sample = "the quick brown fox jumps over a very lazy dog"

    fertilities = []
    for vocab_size in (260, 280, 320):
        merges, _ = train_bpe(corpus, vocab_size)
        fertilities.append(fertility(sample, merges))

    if not (fertilities[0] > fertilities[1] > fertilities[2]):
        fail(
            "fertility should strictly decrease as the vocabulary grows on this "
            f"repetitive corpus, but got {fertilities} for vocab sizes 260, 280, 320"
        )
