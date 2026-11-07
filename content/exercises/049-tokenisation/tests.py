"""Checks for the tokenisation exercise: bytes, byte-pair encoding, and measurement."""

import numpy as np

from mlcheck import check, equal, fail, has


def test_exact_bytes(sub):
    f = has(sub, "text_to_bytes")
    equal(list(f("h")), [104], label="bytes of h")
    equal(list(f("é")), [195, 169], label="bytes of é")
    equal(list(f("世")), [228, 184, 150], label="bytes of 世")
    equal(list(f("🎉")), [240, 159, 142, 137], label="bytes of 🎉")


def test_range_and_length(sub):
    f = has(sub, "text_to_bytes")
    rng = np.random.default_rng(0)
    alphabet = list("abc xyz é世界🎉\n\t—")
    for _ in range(20):
        text = "".join(rng.choice(alphabet, size=int(rng.integers(1, 30))))
        ids = f(text)
        if not all(isinstance(i, int) and 0 <= i <= 255 for i in ids):
            fail(f"every id must be an int from 0 to 255, got {ids} for {text!r}")
        if len(ids) != len(text.encode("utf-8")):
            fail(f"{text!r} has {len(text.encode('utf-8'))} bytes, got {len(ids)} ids")


def test_unknown_worked(sub):
    f = has(sub, "unknown_rate")
    got = f("the cat sat on the mat", {"the", "cat", "sat"})
    if abs(got - 2 / 6) > 1e-12:
        fail(f"expected 2/6 = 0.3333 (on and mat are missing), got {got}")


def test_unknown_extremes(sub):
    f = has(sub, "unknown_rate")
    text = "one fish two fish red fish"
    if f(text, set(text.split())) != 0.0:
        fail("a list holding every word should give an unknown rate of 0.0")
    if f(text, set()) != 1.0:
        fail("an empty list should give an unknown rate of 1.0")


def test_unknown_edges(sub):
    f = has(sub, "unknown_rate")
    got = f("The cat", {"the", "cat"})
    if abs(got - 0.5) > 1e-12:
        fail(f"'The' is not 'the', so expected 0.5, got {got}")
    got = f("", {"a"})
    if got != 0.0:
        fail(f"text with no words should give 0.0, got {got}")
    got = f("  \n ", {"a"})
    if got != 0.0:
        fail(f"whitespace only has no words and should give 0.0, got {got}")


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


def test_merge(sub):
    f = has(sub, "merge_pair")
    same(f([[1, 2, 3, 1, 2], [1, 2]], (1, 2), 9), [[9, 3, 9], [9]], "merge of (1, 2)")
    same(f([[5, 5, 5]], (5, 5), 20), [[20, 5]], "merge of a run of three")
    same(f([[5, 5, 5, 5]], (5, 5), 20), [[20, 20]], "merge of a run of four")
    test_merge_property(sub)


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


def test_tie(sub):
    f = has(sub, "train_bpe")
    merges, _ = f("ab\ncd", 257)
    equal(merges, [(97, 98)], label="first merge when (97, 98) and (99, 100) tie")
    merges, _ = f("cd\nab", 257)
    equal(merges, [(97, 98)], label="first merge does not depend on order")
    test_boundary(sub)


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


def test_encode_raw(sub):
    f = has(sub, "encode")
    for text in ("hello, world", "héllo 世界 🎉", "  two  spaces\tand a tab\n", ""):
        equal(list(f(text, [])), list(text.encode("utf-8")), label=f"bytes of {text!r}")
    test_encode_pieces(sub)


def test_decode(sub):
    f = has(sub, "decode")
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"\xf0\x9f"
    vocab[257] = b"\x8e\x89"
    equal(f([256, 257], vocab), "🎉", label="emoji cut after its second byte")
    equal(f([], vocab), "", label="empty ids")
    equal(f([72, 105], vocab), "Hi", label="plain bytes")


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


def test_decode_bytes_only(sub):
    enc = has(sub, "text_to_bytes")
    dec = has(sub, "decode")
    vocab = {i: bytes([i]) for i in range(256)}
    rng = np.random.default_rng(1)
    alphabet = list("hello wörld 世界 🎉\n")
    for _ in range(20):
        text = "".join(rng.choice(alphabet, size=int(rng.integers(0, 40))))
        equal(dec(enc(text), vocab), text, label=f"bytes-only round trip of {text!r}")


ENGLISH = (
    "the cat sat on the mat. the dog sat on the log. the cat and the dog are friends. "
    "a cat likes to sit in the sun, and a dog likes to run in the park. "
    "the children are playing in the garden, and the teacher is reading a book. "
    "we are learning about the world, and the world is full of interesting things. "
    "the morning was quiet, and the street was empty. nothing was moving in the town. "
    "she was reading, he was writing, and they were singing in the evening. "
    "the running water is shining, and the singing birds are flying over the hills. "
)
HINDI = (
    "बिल्ली चटाई पर बैठी थी। मुझे चाय पसंद है। आज मौसम बहुत अच्छा है। "
    "वह रोज़ सुबह स्कूल जाता है। हम किताबें पढ़ना पसंद करते हैं। कुत्ता बगीचे में दौड़ रहा है। "
    "माँ रसोई में खाना बना रही हैं। बच्चे मैदान में खेल रहे हैं। "
)
EN_SENTENCE = "the cat sat on the mat."
HI_SENTENCE = "बिल्ली चटाई पर बैठी थी।"
SAMPLE = "the dog and the cat were singing in the garden"


def learned(sub, corpus, vocab_size):
    """Train with the reader's module and return only the merge list."""
    return has(sub, "train_bpe")(corpus, vocab_size)[0]


def test_fertility_raw(sub):
    f = has(sub, "fertility")
    got = f("the quick fox", [])
    if abs(got - 13 / 3) > 1e-9:
        fail(f"the quick fox is 13 bytes in 3 words, expected 4.3333, got {got}")
    got = f("hello  world", [])
    if abs(got - 6.0) > 1e-9:
        fail(f"hello  world is 12 bytes in 2 words, expected 6.0, got {got}")
    if abs(f("a", []) - 1.0) > 1e-9:
        fail("a single letter is 1 token in 1 word, so fertility 1.0")


def test_fertility_falls(sub):
    f = has(sub, "fertility")
    merges = learned(sub, ENGLISH, 256 + 120)
    values = [f(SAMPLE, merges[:k]) for k in (0, 20, 80)]
    if not (values[0] > values[1] > values[2]):
        fail(f"fertility should fall strictly with 0, 20, 80 merges, got {values}")
    if values[2] > 2.0:
        fail(f"80 merges on this corpus should bring fertility under 2.0, got {values[2]}")
    unseen = f("zzzz xxxx", merges[:80])
    if abs(unseen - 4.5) > 1e-9:
        fail(f"words made of letters never seen cannot shrink: expected 4.5, got {unseen}")


def test_tpc_raw(sub):
    f = has(sub, "tokens_per_char")
    for text, want in (("abc", 1.0), ("é", 2.0), ("世界", 3.0), ("🎉", 4.0), ("a世", 2.0)):
        got = f(text, [])
        if abs(got - want) > 1e-9:
            fail(f"tokens_per_char({text!r}, []) should be {want}, got {got}")


def test_fairness_gap(sub):
    f = has(sub, "tokens_per_char")
    merges = learned(sub, ENGLISH, 256 + 80)
    en = f(EN_SENTENCE, merges)
    hi = f(HI_SENTENCE, merges)
    if en > 0.5:
        fail(f"English should cost under 0.5 tokens per character after 80 merges, got {en:.3f}")
    if hi < 2.5:
        fail(f"Hindi has no merges of its own and should cost about 2.65 per character, got {hi:.3f}")
    if hi / en < 4.0:
        fail(f"the gap hi / en should be above 4, got {hi / en:.2f}")


def test_shared_budget(sub):
    f = has(sub, "tokens_per_char")
    only_en = learned(sub, ENGLISH, 256 + 80)
    both = learned(sub, ENGLISH + HINDI, 256 + 80)
    if not f(HI_SENTENCE, both) < f(HI_SENTENCE, only_en) / 1.5:
        fail("training on Hindi too should cut its cost per character by a lot")
    if not f(EN_SENTENCE, both) > f(EN_SENTENCE, only_en):
        fail("with a fixed budget, English should cost more once merges go to Hindi")


def test_digit_cuts(sub):
    f = has(sub, "digit_token_lengths")
    whole = learned(sub, "1234\n" * 20, 259)
    apart = learned(sub, "12\n34\n" * 20, 258)
    english = learned(sub, ENGLISH, 256 + 50)
    equal(list(f("1234", whole)), [4], label="trained on 1234")
    equal(list(f("1234", apart)), [2, 2], label="trained on 12 and 34 apart")
    equal(list(f("1234", english)), [1, 1, 1, 1], label="trained on English")
    equal(list(f("no digits here", whole)), [], label="a text without digits")


def test_digit_sums(sub):
    f = has(sub, "digit_token_lengths")
    whole = learned(sub, "1234\n" * 20, 259)
    equal(list(f("11234", whole)), [1, 4], label="11234")
    equal(list(f("12345", whole)), [4, 1], label="12345")
    equal(list(f("price 1234 and 56", whole)), [4, 1, 1], label="price 1234 and 56")
    rng = np.random.default_rng(0)
    alphabet = list("0123456789 ab.\n")
    for _ in range(20):
        text = "".join(rng.choice(alphabet, size=int(rng.integers(1, 30))))
        got = f(text, whole)
        want = sum(ch.isdigit() for ch in text)
        if sum(got) != want:
            fail(f"digit lengths {got} sum to {sum(got)}, but {text!r} has {want} digits")


@check("text_to_bytes gives the real UTF-8 bytes of h, é, 世 and 🎉, as plain ints between 0 and 255")
def check_01(sub):
    test_exact_bytes(sub)
    test_range_and_length(sub)


@check("unknown_rate is the share of words a list misses: exact, case-sensitive, and 0.0 for empty text")
def check_02(sub):
    test_unknown_worked(sub)
    test_unknown_extremes(sub)
    test_unknown_edges(sub)


@check("get_pair_counts counts pairs inside each word, never across words, and counts overlapping runs by position")
def check_03(sub):
    test_counts(sub)


@check("merge_pair glues non-overlapping pairs from the left, keeps its input intact, and never changes what the tokens expand to")
def check_04(sub):
    test_merge(sub)


@check("train_bpe on aaabdaaabac learns the three merges worked by hand, breaks ties to the smaller pair, stays inside words, and stops at the size or when no pair is left")
def check_05(sub):
    test_classic(sub)
    test_tie(sub)
    test_size(sub)


@check("apply_merges_to_word replays the merges in order, and the reversed order gives 7 tokens, not 5")
def check_06(sub):
    test_word(sub)


@check("encode gives raw bytes with no merges and replays the merges inside each piece, never across pieces")
def check_07(sub):
    test_encode_raw(sub)


@check("decode joins the bytes first, so it handles a split character, and decode(encode(text)) is exactly text after training")
def check_08(sub):
    test_decode(sub)
    test_decode_bytes_only(sub)
    test_round_trip(sub)


@check("fertility and tokens_per_char: merges shrink seen words, an English-trained tokeniser overcharges Hindi, and a fixed budget is shared")
def check_09(sub):
    test_fertility_raw(sub)
    test_fertility_falls(sub)
    test_tpc_raw(sub)
    test_fairness_gap(sub)
    test_shared_budget(sub)


@check("digit_token_lengths gives the three cuts of 1234, follows tokens and not place value, and always adds up to the digits")
def check_10(sub):
    test_digit_cuts(sub)
    test_digit_sums(sub)
