"""Byte-level byte-pair encoding: train a merge list, then encode and decode with it.

Everything works on lists of integer ids. The base alphabet is the 256 byte
values, ids 0 to 255. The merge at position i in a merge list always creates
id 256 + i, so the list's order already says what every id means.

`_pretokenize` is provided. It splits text into pieces: a word with at most
one leading space, with punctuation attached to its word, and every run of
whitespace that a leading space cannot absorb as its own piece. Joining the
pieces gives the text back. Merges never cross a piece boundary.

A trained tokeniser is frozen: `encode` replays the merges in the order they
were learned and never counts anything.
"""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Split text into pieces that join back into the exact input."""
    return _PRETOKEN_PATTERN.findall(text)


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


if __name__ == "__main__":
    merges, vocab = train_bpe("aaabdaaabac", 259)
    print("merges:", merges)
    for i, pair in enumerate(merges):
        print(f"  id {256 + i} = {pair} = {vocab[256 + i]!r}")

    ids = encode("aaabdaaabac", merges)
    print("ids:", ids, "(expected [258, 100, 258, 97, 99])")
    print("reversed merge order:", encode("aaabdaaabac", list(reversed(merges))))

    corpus = "the cat sat on the mat. the dog sat on the log. " * 4
    merges, vocab = train_bpe(corpus, 290)
    shown = [vocab[256 + i].decode() for i in range(len(merges))]
    print("English merges:", shown)
    for text in ("the cat sat", "héllo 世界 🎉"):
        ids = encode(text, merges)
        print(f"{text!r}: {len(text.encode('utf-8'))} bytes -> {len(ids)} ids,",
              "round trip ok:", decode(ids, vocab) == text)
