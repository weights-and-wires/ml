"""Tokenisation, from bytes to a measured byte-pair encoder.

You write eleven functions. `text_to_bytes` and `unknown_rate` show why bytes
are the safe alphabet. `get_pair_counts`, `merge_pair`, `train_bpe`,
`apply_merges_to_word`, `encode`, and `decode` are a byte-pair encoder: train
an ordered merge list, replay it to encode, and join bytes to decode.
`fertility`, `tokens_per_char`, and `digit_token_lengths` measure what a
tokeniser charges.

Everything works on lists of integer ids. The base alphabet is the 256 byte
values, ids 0 to 255. The merge at position i in a merge list always creates
id 256 + i, so the list's order already says what every id means.

`_pretokenize` and `build_vocab` are provided. `_pretokenize` splits text into
pieces: a word with at most one leading space, with punctuation attached to
its word, and every run of whitespace that a leading space cannot absorb as
its own piece. Joining the pieces gives the text back. Merges never cross a
piece boundary.
"""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Split text into pieces that join back into the exact input."""
    return _PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# The functions the checks call. Fill these in.
# ---------------------------------------------------------------------------


def text_to_bytes(text: str) -> list[int]:
    """Return the UTF-8 bytes of `text` as a list of ints.

    Args:
        text: Any string.

    Returns:
        A list of ints, each between 0 and 255. Its length is at least
        len(text), because one character can take up to four bytes.
    """
    raise NotImplementedError("text_to_bytes")


def unknown_rate(text: str, word_vocab: set[str]) -> float:
    """Return the share of whitespace-separated words missing from `word_vocab`.

    The match is exact and case-sensitive, so "The" is not "the".

    Args:
        text: The text to check.
        word_vocab: A set of known words.

    Returns:
        missing words divided by total words, a float from 0.0 to 1.0.
        Return 0.0 when `text` has no words.
    """
    raise NotImplementedError("unknown_rate")


def get_pair_counts(token_lists: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count every adjacent pair inside each list.

    Each inner list is scanned on its own, so no pair spans two lists. A list
    that holds the same pair twice adds 2 to that pair's count.

    Args:
        token_lists: A list of lists of ids.

    Returns:
        A dict from (a, b) tuples to counts. Pairs that never occur are absent.
    """
    raise NotImplementedError("get_pair_counts")


def merge_pair(
    token_lists: list[list[int]], pair: tuple[int, int], new_id: int
) -> list[list[int]]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`.

    Scan each inner list from the left. When the current id and the next one
    equal `pair`, write `new_id` and move on by two. Otherwise write the
    current id and move on by one. So [5, 5, 5] with pair (5, 5) gives [20, 5].

    Args:
        token_lists: A list of lists of ids. Do not modify it.
        pair: The two ids to glue.
        new_id: The id that stands for the glued pair.

    Returns:
        A new list of lists, the same length as `token_lists`.
    """
    raise NotImplementedError("merge_pair")


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Learn merges from `corpus` until the vocabulary has `vocab_size` ids.

    Split the corpus with `_pretokenize`, turn each piece into a list of its
    UTF-8 byte values, then repeat: count pairs, take the most frequent one
    (a tie goes to the smaller pair as a tuple), give it the next id, and
    merge it everywhere. Stop at `vocab_size`, or when no pair is left.

    Args:
        corpus: The training text.
        vocab_size: The target number of ids, at least 256.

    Returns:
        (merges, vocab). `merges` lists the pairs in the order learned, and
        merges[i] creates id 256 + i. `vocab` maps every id, 0 to 255 and each
        merged id, to the bytes it stands for.
    """
    raise NotImplementedError("train_bpe")


def apply_merges_to_word(word_bytes: list[int], merges: list[tuple[int, int]]) -> list[int]:
    """Replay `merges`, in order, on the bytes of one piece of text.

    merges[i] creates id 256 + i. Apply merges[0] first, then merges[1], and so
    on, each time gluing the pair from the left with `merge_pair`. Do not count
    pairs, and do not change the order of `merges`.

    Args:
        word_bytes: The UTF-8 bytes of one piece, as a list of ints. Do not modify it.
        merges: The ordered merge list from `train_bpe`.

    Returns:
        The piece as a list of ids after every merge has been tried once.
    """
    raise NotImplementedError("apply_merges_to_word")


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Turn `text` into one flat list of ids.

    Split `text` with `_pretokenize`, turn each piece into its UTF-8 bytes,
    replay the merges on each piece with `apply_merges_to_word`, and join the
    results in order. With an empty `merges`, the result is the raw bytes.

    Args:
        text: Any string.
        merges: The ordered merge list.

    Returns:
        A flat list of ids.
    """
    raise NotImplementedError("encode")


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Turn a list of ids back into text.

    Look up the bytes of every id, join all of them, and decode UTF-8 once. One
    token can hold half of a character, so never decode a token on its own.

    Args:
        ids: A list of ids, as `encode` returns.
        vocab: A dict from id to bytes, as `train_bpe` returns.

    Returns:
        The decoded string.
    """
    raise NotImplementedError("decode")


# ---------------------------------------------------------------------------
# Provided for the measurements: the bytes of every id.
# ---------------------------------------------------------------------------


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """Return the bytes of every id: the 256 byte values, then one entry per merge."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return vocab


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    """Return the average number of tokens per word.

    A word is a whitespace-separated item, counted with `text.split()`. That
    count is separate from the pieces `_pretokenize` makes.

    Args:
        text: A string with at least one word.
        merges: The ordered merge list.

    Returns:
        len(encode(text, merges)) divided by the number of words, as a float.
    """
    raise NotImplementedError("fertility")


def tokens_per_char(text: str, merges: list[tuple[int, int]]) -> float:
    """Return the average number of tokens per character.

    Characters are what `len(text)` counts, not bytes. With no merges, a
    character costs as many tokens as it has UTF-8 bytes.

    Args:
        text: A non-empty string.
        merges: The ordered merge list.

    Returns:
        len(encode(text, merges)) divided by len(text), as a float.
    """
    raise NotImplementedError("tokens_per_char")


def digit_token_lengths(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Return how many ASCII digits each digit-holding token covers.

    Encode `text`, look up each token's bytes with `build_vocab(merges)`, and
    count the bytes between 48 and 57 (the characters 0 to 9). Keep the count
    for every token that has at least one digit, in order. Tokens with no
    digits are skipped.

    Args:
        text: Any string.
        merges: The ordered merge list.

    Returns:
        A list of ints. Its sum is the number of ASCII digits in `text`.
    """
    raise NotImplementedError("digit_token_lengths")


if __name__ == "__main__":
    sample = "héllo 世界 🎉"
    ids = text_to_bytes(sample)
    print(f"{sample!r}: {len(sample)} characters, {len(ids)} bytes")
    word_vocab = {"the", "cat", "sat"}
    text = "the cat sat on the mat"
    print(f"unknown rate of a 3-word list on {text!r}: {unknown_rate(text, word_vocab):.2f}")
    n = 5700
    print(f"attention pairs for {n} bytes: {n * n:,}; for 1000 words: {1000 * 1000:,}")

    merges, vocab = train_bpe("aaabdaaabac", 259)
    print("\nmerges:", merges)
    for i, pair in enumerate(merges):
        print(f"  id {256 + i} = {pair} = {vocab[256 + i]!r}")
    print("ids:", encode("aaabdaaabac", merges), "(expected [258, 100, 258, 97, 99])")
    print("reversed merge order:", encode("aaabdaaabac", list(reversed(merges))))

    english = (
        "the cat sat on the mat. the dog sat on the log. the cat and the dog are friends. "
        "a cat likes to sit in the sun, and a dog likes to run in the park. "
        "the children are playing in the garden, and the teacher is reading a book. "
    )
    en_sentence = "the cat sat on the mat."
    hi_sentence = "बिल्ली चटाई पर बैठी थी।"
    merges, vocab = train_bpe(english, 256 + 60)
    for text in (en_sentence, "héllo 世界 🎉"):
        ids = encode(text, merges)
        print(f"{text!r}: {len(text.encode('utf-8'))} bytes -> {len(ids)} ids,",
              "round trip ok:", decode(ids, vocab) == text)

    print("\nfertility of 'the quick fox' with no merges:", round(fertility("the quick fox", []), 2))
    for name, text in (("English", en_sentence), ("Hindi", hi_sentence)):
        print(f"{name:8s} tokens per character: {tokens_per_char(text, merges):.2f}")
    digits_merges, _ = train_bpe("1234\n" * 20, 259)
    print("cuts of 1234 after training on 1234:", digit_token_lengths("1234", digits_merges))
    print("cuts of 1234 with no merges:", digit_token_lengths("1234", []))
