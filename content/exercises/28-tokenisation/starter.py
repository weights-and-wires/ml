"""Byte-level byte-pair encoding, built from its two primitives up.

Everything operates on lists of integer symbol IDs. The base alphabet is the
256 possible byte values, IDs 0 through 255. Every merge you (or `train_bpe`)
perform introduces one new, larger ID for a pair of existing symbols, and the
convention this file uses throughout is that the merge at position i in a
merges list always introduces ID 256 + i, so you never need to invent or track
IDs separately from a merge list's position.

`_pretokenize` below is provided and used by both `train_bpe` and `encode`.
It splits text into spans on whitespace boundaries, attaching a single
leading space to the word that follows it and treating any run of two or
more whitespace characters as its own span, so concatenating the spans
reproduces the input exactly. Merges never cross a span boundary. You do not
need to change it.
"""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Split text into whitespace-respecting spans that concatenate back exactly."""
    return _PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# The six functions the checks call. Fill these in.
# ---------------------------------------------------------------------------


def get_pair_counts(token_lists: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count every adjacent pair of symbols across a list of token-ID lists.

    Each inner list is scanned independently; pairs never span across two
    different inner lists. A word contributing the same adjacent pair twice
    adds 2 to that pair's count.

    Args:
        token_lists: A list of lists of symbol IDs.

    Returns:
        A dict mapping each adjacent pair (a, b), as a 2-tuple of ints, to
        the number of times it occurs. Pairs that never occur are absent
        from the dict rather than mapped to 0.
    """
    raise NotImplementedError("get_pair_counts")


def merge_pair(
    token_lists: list[list[int]], pair: tuple[int, int], new_id: int
) -> list[list[int]]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`.

    Scan each inner list left to right. When the symbol at the current
    position and the next one equal `pair`, emit `new_id` and advance by two
    positions; otherwise emit the current symbol and advance by one. This
    matches the standard BPE merge: overlapping occurrences of `pair`
    (as in three identical symbols in a row, when `pair` is that symbol
    repeated) only merge every other one, left to right.

    Args:
        token_lists: A list of lists of symbol IDs.
        pair: The 2-tuple of symbol IDs to merge.
        new_id: The symbol ID that replaces every merged occurrence.

    Returns:
        A new list of lists, the same length as `token_lists`, with `pair`
        merged wherever it occurred. Does not need to mutate its input.
    """
    raise NotImplementedError("merge_pair")


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Train a byte-level BPE tokeniser on `corpus` up to `vocab_size` symbols.

    Split `corpus` with `_pretokenize`, encode each resulting span to its
    UTF-8 bytes, and represent each span as the list of those byte values.
    Repeatedly find the most frequent adjacent pair across every span (with
    `get_pair_counts`), breaking ties by choosing the pair that is smaller as
    a tuple of IDs, assign it the next symbol ID (256 for the first merge,
    257 for the second, and so on), and replace it everywhere (with
    `merge_pair`). Stop once the vocabulary reaches `vocab_size` symbols, or
    once no pair remains to merge, whichever comes first.

    Args:
        corpus: The training text.
        vocab_size: The target vocabulary size, at least 256.

    Returns:
        A tuple (merges, vocab):
            merges: The ordered list of merges performed, each a 2-tuple of
                the symbol IDs that were merged. merges[i] always introduces
                symbol ID 256 + i.
            vocab: A dict mapping every symbol ID, 0 through 255 and every
                merged ID, to the bytes it stands for. vocab[i] for i < 256
                is `bytes([i])`; vocab[256 + i] is the concatenation of the
                bytes of the two symbols merges[i] combined.
    """
    raise NotImplementedError("train_bpe")


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Encode text into symbol IDs by applying `merges`, in order.

    Split `text` with `_pretokenize` and encode each span to a list of its
    UTF-8 byte values. Apply merges[0] to every span, then merges[1], and so
    on, in the order given, never by recomputing frequencies on `text`
    itself. merges[i] always introduces symbol ID 256 + i, whether or not
    `merges` is the list `train_bpe` produced. Concatenate every span's
    final symbol list, in span order, into one flat list.

    Args:
        text: The text to encode.
        merges: An ordered list of pairs to merge, as `train_bpe` returns.

    Returns:
        A flat list of symbol IDs for the whole of `text`.
    """
    raise NotImplementedError("encode")


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decode a list of symbol IDs back into text.

    Args:
        ids: A list of symbol IDs, as `encode` returns.
        vocab: A dict mapping symbol ID to the bytes it stands for, as
            `train_bpe` returns.

    Returns:
        The decoded string.
    """
    raise NotImplementedError("decode")


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    """Return the average number of tokens per word `encode` produces for text.

    A "word" here is counted by `len(text.split())`, a plain whitespace
    split, independent of `_pretokenize`'s span convention.

    Args:
        text: The text to measure.
        merges: An ordered list of pairs to merge, as `train_bpe` returns.

    Returns:
        `len(encode(text, merges)) / len(text.split())`, as a float. Assume
        `text` has at least one word.
    """
    raise NotImplementedError("fertility")


if __name__ == "__main__":
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox runs from the lazy dog. "
    ) * 20

    for vocab_size in (260, 280, 320):
        merges, vocab = train_bpe(corpus, vocab_size)
        sample = "the quick brown fox jumps over a very lazy dog"
        ids = encode(sample, merges)
        back = decode(ids, vocab)
        assert back == sample, "decode(encode(text)) must reproduce text exactly"
        print(
            f"vocab_size={vocab_size:4d}  merges={len(merges):3d}  "
            f"tokens for sample={len(ids):3d}  fertility={fertility(sample, merges):.2f}"
        )

    # Byte-level tokenisation never fails on text the corpus never saw,
    # including scripts and symbols entirely absent from training.
    unseen = "héllo — 世界 🎉"
    merges, vocab = train_bpe(corpus, 270)
    assert decode(encode(unseen, merges), vocab) == unseen
    print(f"round-tripped text the training corpus never saw: {unseen!r}")
