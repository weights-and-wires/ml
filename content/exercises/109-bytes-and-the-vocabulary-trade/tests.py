"""Checks for the bytes and vocabulary exercise."""

import numpy as np

from mlcheck import check, equal, fail, has


@check("text_to_bytes gives the real UTF-8 bytes of h, é, 世 and 🎉")
def test_exact_bytes(sub):
    f = has(sub, "text_to_bytes")
    equal(list(f("h")), [104], label="bytes of h")
    equal(list(f("é")), [195, 169], label="bytes of é")
    equal(list(f("世")), [228, 184, 150], label="bytes of 世")
    equal(list(f("🎉")), [240, 159, 142, 137], label="bytes of 🎉")


@check("text_to_bytes returns plain ints between 0 and 255, longer than the string when needed")
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


@check("bytes_to_text reverses text_to_bytes for any text, with no unknowns")
def test_round_trip(sub):
    enc = has(sub, "text_to_bytes")
    dec = has(sub, "bytes_to_text")
    rng = np.random.default_rng(1)
    alphabet = list("hello wörld 世界 🎉\n")
    for _ in range(20):
        text = "".join(rng.choice(alphabet, size=int(rng.integers(0, 40))))
        equal(dec(enc(text)), text, label=f"round trip of {text!r}")
    equal(dec([240, 159, 142, 137]), "🎉", label="four bytes decoded together")


@check("unknown_rate on the worked example is 2 of 6 words")
def test_unknown_worked(sub):
    f = has(sub, "unknown_rate")
    got = f("the cat sat on the mat", {"the", "cat", "sat"})
    if abs(got - 2 / 6) > 1e-12:
        fail(f"expected 2/6 = 0.3333 (on and mat are missing), got {got}")


@check("unknown_rate is 0 for a full list and 1 for an empty list")
def test_unknown_extremes(sub):
    f = has(sub, "unknown_rate")
    text = "one fish two fish red fish"
    if f(text, set(text.split())) != 0.0:
        fail("a list holding every word should give an unknown rate of 0.0")
    if f(text, set()) != 1.0:
        fail("an empty list should give an unknown rate of 1.0")


@check("unknown_rate is exact and case-sensitive, and handles empty text")
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
