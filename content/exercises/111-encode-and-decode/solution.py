"""Encode text with a learned merge list, and decode ids back into text.

The merge at position i in a merge list creates id 256 + i, and the ids 0 to
255 are the byte values. A trained tokeniser is frozen: encoding replays the
merges in the order they were learned and never counts anything.
"""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Split text into pieces that join back into the exact input."""
    return _PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Written in the chapter "Training byte-pair encoding: merge the most common
# pair". They are solved here so you can use them. Do not change them.
# ---------------------------------------------------------------------------


def get_pair_counts(token_lists: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count every adjacent pair inside each list."""
    counts: dict[tuple[int, int], int] = {}
    for tokens in token_lists:
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(
    token_lists: list[list[int]], pair: tuple[int, int], new_id: int
) -> list[list[int]]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`, from the left."""
    result = []
    for tokens in token_lists:
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        result.append(merged)
    return result


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Learn an ordered merge list and the vocabulary of bytes for every id."""
    token_lists = [list(piece.encode("utf-8")) for piece in _pretokenize(corpus)]
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[int, int]] = []
    while 256 + len(merges) < vocab_size:
        counts = get_pair_counts(token_lists)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + len(merges)
        merges.append(best)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        token_lists = merge_pair(token_lists, best, new_id)
    return merges, vocab


# ---------------------------------------------------------------------------
# Written in this chapter.
# ---------------------------------------------------------------------------


def apply_merges_to_word(word_bytes: list[int], merges: list[tuple[int, int]]) -> list[int]:
    ids = list(word_bytes)
    for i, pair in enumerate(merges):
        ids = merge_pair([ids], pair, 256 + i)[0]
    return ids


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids: list[int] = []
    for piece in _pretokenize(text):
        ids.extend(apply_merges_to_word(list(piece.encode("utf-8")), merges))
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8")


if __name__ == "__main__":
    merges, vocab = train_bpe("aaabdaaabac", 259)
    ids = encode("aaabdaaabac", merges)
    print("ids:", ids, "(expected [258, 100, 258, 97, 99])")

    reversed_ids = encode("aaabdaaabac", list(reversed(merges)))
    print("reversed merge order:", reversed_ids)

    corpus = "the cat sat on the mat. the dog sat on the log. " * 4
    merges, vocab = train_bpe(corpus, 290)
    for text in ("the cat sat", "héllo 世界 🎉"):
        ids = encode(text, merges)
        print(f"{text!r}: {len(text.encode('utf-8'))} bytes -> {len(ids)} ids,",
              "round trip ok:", decode(ids, vocab) == text)
