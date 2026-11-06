"""Measure a tokeniser: tokens per word, tokens per character, and digit cuts.

The merge at position i in a merge list creates id 256 + i. `train_bpe`,
`encode`, and `decode` come from the earlier chapters and are already solved.
"""

import re

_PRETOKEN_PATTERN = re.compile(r" ?[^\s]+|\s+")


def _pretokenize(text: str) -> list[str]:
    """Split text into pieces that join back into the exact input."""
    return _PRETOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Written in the chapter "Byte-pair encoding: train it, then use it".
# Solved here so you can use them. Do not change them.
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
# Written in the same chapter. Solved here so you can use them. Do not change them.
# ---------------------------------------------------------------------------


def apply_merges_to_word(word_bytes: list[int], merges: list[tuple[int, int]]) -> list[int]:
    """Replay the merges, in order, on the bytes of one piece."""
    ids = list(word_bytes)
    for i, pair in enumerate(merges):
        ids = merge_pair([ids], pair, 256 + i)[0]
    return ids


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Turn text into one flat list of ids."""
    ids: list[int] = []
    for piece in _pretokenize(text):
        ids.extend(apply_merges_to_word(list(piece.encode("utf-8")), merges))
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Join the bytes of every id, then decode UTF-8 once."""
    return b"".join(vocab[i] for i in ids).decode("utf-8")


# ---------------------------------------------------------------------------
# Provided for this chapter.
# ---------------------------------------------------------------------------


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """Return the bytes of every id: the 256 byte values, then one entry per merge."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab[256 + i] = vocab[a] + vocab[b]
    return vocab


# ---------------------------------------------------------------------------
# Written in this chapter.
# ---------------------------------------------------------------------------


def fertility(text: str, merges: list[tuple[int, int]]) -> float:
    return len(encode(text, merges)) / len(text.split())


def tokens_per_char(text: str, merges: list[tuple[int, int]]) -> float:
    return len(encode(text, merges)) / len(text)


def digit_token_lengths(text: str, merges: list[tuple[int, int]]) -> list[int]:
    vocab = build_vocab(merges)
    lengths = []
    for i in encode(text, merges):
        digits = sum(1 for byte in vocab[i] if 48 <= byte <= 57)
        if digits > 0:
            lengths.append(digits)
    return lengths


if __name__ == "__main__":
    english = (
        "the cat sat on the mat. the dog sat on the log. the cat and the dog are friends. "
        "a cat likes to sit in the sun, and a dog likes to run in the park. "
        "the children are playing in the garden, and the teacher is reading a book. "
    )
    en_sentence = "the cat sat on the mat."
    hi_sentence = "बिल्ली चटाई पर बैठी थी।"
    merges, _ = train_bpe(english, 256 + 60)

    print("fertility of 'the quick fox' with no merges:", round(fertility("the quick fox", []), 2))
    for name, text in (("English", en_sentence), ("Hindi", hi_sentence)):
        print(f"{name:8s} tokens per character: {tokens_per_char(text, merges):.2f}")

    digits_merges, _ = train_bpe("1234\n" * 20, 259)
    print("cuts of 1234 after training on 1234:", digit_token_lengths("1234", digits_merges))
    print("cuts of 1234 with no merges:", digit_token_lengths("1234", []))
