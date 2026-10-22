"""Checks for the encode and decode exercise."""

import numpy as np

from mlcheck import check, equal, fail, has

CLASSIC = [(97, 97), (97, 98), (256, 257)]
ALPHABET = list("the cat sat on a mat, éè 世界 🎉\n\t.")


@check("apply_merges_to_word replays the merges in order on one piece")
def test_word(sub):
    f = has(sub, "apply_merges_to_word")
    equal(list(f([97, 97, 97, 98], CLASSIC)), [258], label="aaab")
    got = list(f(list(b"aaabdaaabac"), CLASSIC))
    equal(got, [258, 100, 258, 97, 99], label="aaabdaaabac")
    equal(list(f([104, 105], [])), [104, 105], label="no merges")


@check("apply_merges_to_word leaves its input alone and skips a merge whose ids do not exist yet")
def test_word_edges(sub):
    f = has(sub, "apply_merges_to_word")
    word = [97, 97, 97, 98]
    f(word, CLASSIC)
    equal(word, [97, 97, 97, 98], label="the input list")
    got = list(f([97, 98], [(256, 257), (97, 98)]))
    equal(got, [257], label="a merge on ids that do not exist is idle")


@check("The order of the merges matters: the reversed list gives 7 tokens, not 5")
def test_order(sub):
    f = has(sub, "apply_merges_to_word")
    word = list(b"aaabdaaabac")
    equal(list(f(word, list(reversed(CLASSIC)))), [258, 257, 100, 258, 257, 97, 99], label="reversed order")
    if len(f(word, list(reversed(CLASSIC)))) <= len(f(word, CLASSIC)):
        fail("the reversed order should give more tokens than the learned order")


@check("encode with no merges gives the raw UTF-8 bytes, spaces and all")
def test_encode_raw(sub):
    f = has(sub, "encode")
    for text in ("hello, world", "héllo 世界 🎉", "  two  spaces\tand a tab\n", ""):
        equal(list(f(text, [])), list(text.encode("utf-8")), label=f"bytes of {text!r}")


@check("encode replays the merges inside each piece and never across pieces")
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
